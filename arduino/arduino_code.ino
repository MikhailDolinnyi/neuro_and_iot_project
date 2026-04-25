#include <Wire.h>
#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include "MAX30105.h"
#include "heartRate.h"

// =========================
// Wi-Fi
// =========================
const char* WIFI_SSID = "Гога";
const char* WIFI_PASS = "ggnebudet";

// =========================
// MQTT
// =========================
const char* MQTT_HOST = "172.20.10.2";
const uint16_t MQTT_PORT = 1883;
const char* MQTT_USER = "";
const char* MQTT_PASS = "";

// Один общий топик
const char* TOPIC_DATA = "sensor/health/data";
const char* MQTT_WILL_PAYLOAD = "{\"device\":\"offline\",\"state\":\"offline\"}";

// =========================
// MAX30102
// =========================
MAX30105 particleSensor;
const uint8_t MAX30102_ADDR = 0x57;

bool max30102Ready = false;
unsigned long lastMaxRetry = 0;
const unsigned long MAX30102_RETRY_MS = 2000;

// BPM smoothing
const byte RATE_SIZE = 4;
byte rates[RATE_SIZE];
byte rateSpot = 0;
byte validSamples = 0;

unsigned long lastBeat = 0;
unsigned long lastValidBpmAt = 0;
float beatsPerMinute = 0.0;
float lastCurrentBpm = 0.0;
int beatAvg = 0;

const unsigned long BPM_STALE_MS = 8000;

// Быстрая первая публикация BPM
bool firstBpmPublished = false;
unsigned long lastBpmPublish = 0;
const unsigned long MIN_BPM_PUBLISH_GAP_MS = 1000;

// Порог захвата/потери пальца
const long FINGER_ON_THRESHOLD  = 20000;
const long FINGER_OFF_THRESHOLD = 12000;

const byte FINGER_ON_STABLE_COUNT  = 2;
const byte FINGER_OFF_STABLE_COUNT = 20;

const unsigned long REACQUIRE_WARMUP_MS = 250;

bool fingerDetected = false;
byte fingerOnCounter = 0;
byte fingerOffCounter = 0;
unsigned long fingerAcquiredAt = 0;

// Последний IR
long lastIrValue = 0;

// =========================
// DS18B20
// =========================
#define ONE_WIRE_BUS D5

OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature tempSensor(&oneWire);

bool tempConversionPending = false;
unsigned long tempRequestStartedAt = 0;
const unsigned long DS18B20_CONVERSION_MS = 750;

float lastTempC = 0.0;
bool lastTempValid = false;

// =========================
// FSR402
// =========================
#define FSR_PIN A0
const int FSR_PRESSED_THRESHOLD = 120;

// =========================
// Тайминги
// =========================
unsigned long lastPublish = 0;
const unsigned long PUBLISH_INTERVAL_MS = 3000;

// Ограничение спама в Serial
unsigned long lastStatusSerialLog = 0;
const unsigned long SERIAL_STATUS_INTERVAL_MS = 500;

// =========================
// Сеть
// =========================
WiFiClient espClient;
PubSubClient mqttClient(espClient);

// =========================
// Helpers
// =========================
bool shouldLogStatusNow() {
  return (millis() - lastStatusSerialLog) >= SERIAL_STATUS_INTERVAL_MS;
}

void markStatusLogPrinted() {
  lastStatusSerialLog = millis();
}

void resetStatusLogTimer() {
  lastStatusSerialLog = 0;
}

// =========================
// BPM helpers
// =========================
void addBpmSample(byte bpm) {
  rates[rateSpot++] = bpm;
  rateSpot %= RATE_SIZE;

  if (validSamples < RATE_SIZE) {
    validSamples++;
  }

  int sum = 0;
  for (byte i = 0; i < validSamples; i++) {
    sum += rates[i];
  }
  beatAvg = sum / validSamples;
}

void resetBpmBuffer() {
  rateSpot = 0;
  validSamples = 0;
  beatAvg = 0;
  beatsPerMinute = 0.0;
  lastCurrentBpm = 0.0;
  lastBeat = 0;
  lastValidBpmAt = 0;
  firstBpmPublished = false;
  lastBpmPublish = 0;

  for (byte i = 0; i < RATE_SIZE; i++) {
    rates[i] = 0;
  }
}

void resetFingerState() {
  fingerDetected = false;
  fingerOnCounter = 0;
  fingerOffCounter = 0;
  fingerAcquiredAt = 0;
}

bool isWarmupActive() {
  return fingerDetected && (millis() - fingerAcquiredAt < REACQUIRE_WARMUP_MS);
}

bool isBpmValid() {
  if (!max30102Ready) return false;
  if (!fingerDetected) return false;
  if (isWarmupActive()) return false;
  if (validSamples == 0) return false;
  if (lastValidBpmAt == 0) return false;
  if (millis() - lastValidBpmAt > BPM_STALE_MS) return false;
  return true;
}

bool updateFingerState(long irValue) {
  if (!fingerDetected) {
    if (irValue >= FINGER_ON_THRESHOLD) {
      if (fingerOnCounter < FINGER_ON_STABLE_COUNT) {
        fingerOnCounter++;
      }
    } else {
      fingerOnCounter = 0;
    }

    fingerOffCounter = 0;

    if (fingerOnCounter >= FINGER_ON_STABLE_COUNT) {
      fingerDetected = true;
      fingerAcquiredAt = millis();
      resetBpmBuffer();
      resetStatusLogTimer();
      Serial.println("Finger acquired");
    }
  } else {
    if (irValue <= FINGER_OFF_THRESHOLD) {
      if (fingerOffCounter < FINGER_OFF_STABLE_COUNT) {
        fingerOffCounter++;
      }
    } else {
      fingerOffCounter = 0;
    }

    fingerOnCounter = 0;

    if (fingerOffCounter >= FINGER_OFF_STABLE_COUNT) {
      fingerDetected = false;
      resetBpmBuffer();
      resetStatusLogTimer();
      Serial.println("Finger lost");
    }
  }

  return fingerDetected;
}

// =========================
// MAX30102 helpers
// =========================
bool isMAX30102Present() {
  Wire.beginTransmission(MAX30102_ADDR);
  return (Wire.endTransmission() == 0);
}

bool initMAX30102() {
  if (!isMAX30102Present()) {
    return false;
  }

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    return false;
  }

  byte ledBrightness = 0x1F;
  byte sampleAverage = 4;
  byte ledMode = 2;
  int sampleRate = 400;
  int pulseWidth = 411;
  int adcRange = 16384;

  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
  particleSensor.setPulseAmplitudeRed(0x1F);
  particleSensor.setPulseAmplitudeGreen(0);

  resetBpmBuffer();
  resetFingerState();
  resetStatusLogTimer();
  lastIrValue = 0;

  Serial.println("MAX30102 initialized.");
  return true;
}

void handleMAX30102Lost() {
  if (!max30102Ready) {
    return;
  }

  max30102Ready = false;
  resetBpmBuffer();
  resetFingerState();
  resetStatusLogTimer();
  lastIrValue = 0;

  Serial.println("MAX30102 disconnected. Waiting for reconnect...");
}

void tryReconnectMAX30102() {
  if (max30102Ready) {
    return;
  }

  if (millis() - lastMaxRetry < MAX30102_RETRY_MS) {
    return;
  }

  lastMaxRetry = millis();

  Serial.println("Trying to reconnect MAX30102...");
  Wire.begin(D2, D1);
  delay(20);

  if (initMAX30102()) {
    max30102Ready = true;
    Serial.println("MAX30102 reconnected.");
  } else {
    Serial.println("MAX30102 still not detected.");
  }
}

// =========================
// DS18B20 helpers
// =========================
void startTemperatureConversion() {
  tempSensor.requestTemperatures();
  tempRequestStartedAt = millis();
  tempConversionPending = true;
}

void serviceTemperatureSensor() {
  if (tempConversionPending && (millis() - tempRequestStartedAt >= DS18B20_CONVERSION_MS)) {
    float tempC = tempSensor.getTempCByIndex(0);

    if (tempC != DEVICE_DISCONNECTED_C) {
      lastTempC = tempC;
      lastTempValid = true;
    } else {
      lastTempValid = false;
    }

    tempConversionPending = false;
  }

  if (!tempConversionPending) {
    startTemperatureConversion();
  }
}

// =========================
// FSR helpers
// =========================
int readFsrAveraged() {
  long sum = 0;
  const int samples = 10;

  for (int i = 0; i < samples; i++) {
    sum += analogRead(FSR_PIN);
    delay(2);
  }

  return sum / samples;
}

// =========================
// State helpers
// =========================
const char* getStateString() {
  if (!max30102Ready) {
    return "sensor_missing";
  }

  if (!fingerDetected) {
    return "no_finger";
  }

  return "measuring";
}

// =========================
// MQTT publish
// =========================
void publishUnifiedSnapshot(bool force = false) {
  if (!mqttClient.connected()) {
    return;
  }

  int fsrRaw = readFsrAveraged();
  bool fsrPressed = fsrRaw >= FSR_PRESSED_THRESHOLD;

  bool warmup = isWarmupActive();
  bool bpmValid = isBpmValid();

  char payload[256];

  snprintf(
    payload,
    sizeof(payload),
    "{\"device\":\"online\",\"state\":\"%s\",\"max30102_ready\":%s,\"finger\":%s,\"warmup\":%s,\"ir\":%ld,\"bpm_valid\":%s,\"bpm_current\":%.1f,\"bpm_avg\":%d,\"bpm_samples\":%d,\"temp_valid\":%s,\"temp_c\":%.2f,\"fsr_raw\":%d,\"fsr_pressed\":%s}",
    getStateString(),
    max30102Ready ? "true" : "false",
    fingerDetected ? "true" : "false",
    warmup ? "true" : "false",
    lastIrValue,
    bpmValid ? "true" : "false",
    bpmValid ? lastCurrentBpm : 0.0,
    bpmValid ? beatAvg : 0,
    validSamples,
    lastTempValid ? "true" : "false",
    lastTempValid ? lastTempC : 0.0,
    fsrRaw,
    fsrPressed ? "true" : "false"
  );

  mqttClient.publish(TOPIC_DATA, payload, true);

  if (force) {
    lastPublish = millis();
  }

  Serial.print("MQTT publish: ");
  Serial.println(payload);
}

void tryPublishBpmImmediately() {
  if (!mqttClient.connected()) {
    return;
  }

  if (millis() - lastBpmPublish < MIN_BPM_PUBLISH_GAP_MS) {
    return;
  }

  if (!firstBpmPublished && validSamples >= 1) {
    publishUnifiedSnapshot(true);
    firstBpmPublished = true;
    lastBpmPublish = millis();

    Serial.println("Immediate first BPM publish done");
  }
}

// =========================
// Wi-Fi / MQTT
// =========================
void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.print("WiFi connected, IP: ");
  Serial.println(WiFi.localIP());
}

void connectMQTT() {
  while (!mqttClient.connected()) {
    Serial.print("Connecting to MQTT... ");

    String clientId = "NodeMCU-" + String(ESP.getChipId(), HEX);

    bool connected;
    if (strlen(MQTT_USER) > 0) {
      connected = mqttClient.connect(
        clientId.c_str(),
        MQTT_USER,
        MQTT_PASS,
        TOPIC_DATA,
        0,
        true,
        MQTT_WILL_PAYLOAD
      );
    } else {
      connected = mqttClient.connect(
        clientId.c_str(),
        TOPIC_DATA,
        0,
        true,
        MQTT_WILL_PAYLOAD
      );
    }

    if (connected) {
      Serial.println("connected");
      publishUnifiedSnapshot(true);
    } else {
      Serial.print("failed, rc=");
      Serial.print(mqttClient.state());
      Serial.println(" retry in 2 sec");
      delay(2000);
    }
  }
}

// =========================
// Sensors setup
// =========================
void setupDS18B20() {
  tempSensor.begin();
  tempSensor.setWaitForConversion(false);
  startTemperatureConversion();
  Serial.println("DS18B20 initialized.");
}

// =========================
// Setup
// =========================
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("Booting...");

  Wire.begin(D2, D1);
  setupDS18B20();

  max30102Ready = initMAX30102();
  if (!max30102Ready) {
    Serial.println("MAX30102 not found at startup. Will retry in loop.");
  }

  connectWiFi();
  mqttClient.setServer(MQTT_HOST, MQTT_PORT);
  connectMQTT();

  Serial.println("System ready.");
}

// =========================
// Loop
// =========================
void loop() {
  serviceTemperatureSensor();

  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }

  if (!mqttClient.connected()) {
    connectMQTT();
  }
  mqttClient.loop();

  if (max30102Ready && !isMAX30102Present()) {
    handleMAX30102Lost();
    publishUnifiedSnapshot(true);
  }

  if (!max30102Ready) {
    tryReconnectMAX30102();

    if (shouldLogStatusNow()) {
      Serial.println("MAX30102 missing");
      markStatusLogPrinted();
    }

    if (millis() - lastPublish >= PUBLISH_INTERVAL_MS) {
      publishUnifiedSnapshot(true);
    }

    delay(20);
    return;
  }

  lastIrValue = particleSensor.getIR();

  bool prevFingerPresent = fingerDetected;
  bool fingerPresent = updateFingerState(lastIrValue);

  if (fingerPresent != prevFingerPresent) {
    publishUnifiedSnapshot(true);
  }

  if (!fingerPresent) {
    if (shouldLogStatusNow()) {
      Serial.print("IR=");
      Serial.print(lastIrValue);
      Serial.println(" No finger");
      markStatusLogPrinted();
    }

    if (millis() - lastPublish >= PUBLISH_INTERVAL_MS) {
      publishUnifiedSnapshot(true);
    }

    delay(20);
    return;
  }

  if (isWarmupActive()) {
    if (shouldLogStatusNow()) {
      Serial.print("IR=");
      Serial.print(lastIrValue);
      Serial.println(" Warmup after reacquire");
      markStatusLogPrinted();
    }

    if (millis() - lastPublish >= PUBLISH_INTERVAL_MS) {
      publishUnifiedSnapshot(true);
    }

    delay(20);
    return;
  }

  if (checkForBeat(lastIrValue)) {
    unsigned long now = millis();

    if (lastBeat > 0) {
      unsigned long delta = now - lastBeat;

      if (delta > 0) {
        beatsPerMinute = 60.0 / (delta / 1000.0);

        if (beatsPerMinute > 20 && beatsPerMinute < 255) {
          addBpmSample((byte)beatsPerMinute);

          lastCurrentBpm = beatsPerMinute;
          lastValidBpmAt = now;

          Serial.print("IR=");
          Serial.print(lastIrValue);
          Serial.print(", BPM=");
          Serial.print(lastCurrentBpm);
          Serial.print(", Avg BPM=");
          Serial.print(beatAvg);

          if (validSamples < RATE_SIZE) {
            Serial.print(" [samples ");
            Serial.print(validSamples);
            Serial.print("/");
            Serial.print(RATE_SIZE);
            Serial.print("]");
          }

          Serial.println();

          tryPublishBpmImmediately();
        }
      }
    }

    lastBeat = now;
  }

  if (millis() - lastPublish >= PUBLISH_INTERVAL_MS) {
    publishUnifiedSnapshot(true);
    lastBpmPublish = millis();
  }

  delay(20);
}