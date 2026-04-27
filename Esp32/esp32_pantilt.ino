#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include <Preferences.h>
#include <ESP32Servo.h>
#include <ArduinoJson.h>
#include <math.h>

Preferences prefs;
WebServer server(80);
Servo panServo;
Servo tiltServo;

// =============================================================
// SERVO SETTINGS
// =============================================================
// Если механика позволяет, можно поставить 0 и 180.
// Но для pan-tilt обычно лучше не упираться в крайние положения.
const float SERVO_MIN_ANGLE = 5.0f;
const float SERVO_MAX_ANGLE = 175.0f;

// Подбери под свои серво.
// Если серво упирается, гудит или дергается на краях — попробуй 600..2300.
const int SERVO_MIN_US = 500;
const int SERVO_MAX_US = 2400;

// Частота обновления расчета движения. Серво все равно работает на 50 Гц,
// но расчеты можно делать чаще для более ровного движения.
const unsigned long SERVO_UPDATE_MS = 10;

// Главная скорость обычного движения.
// Если слишком резко — уменьши до 220..260.
// Если медленно — увеличь до 380..450, но дешевые серво могут не успевать.
const float SERVO_SPEED_DPS = 320.0f;

// Базовая скорость при удержании направления /api/start_move?direction=...
// Дальше применяется режим slow/medium/fast.
const float CONTINUOUS_TARGET_SPEED_DPS = 220.0f;

// =============================================================
// CONFIG
// =============================================================
struct DeviceConfig {
  String wifiSsid;
  String wifiPassword;
  String hostName = "trackingcat-pantilt";

  int panPin = 17;
  int tiltPin = 18;
  int laserPin = 5;

  // Целевые углы. Фактические углы panCurrentAngle/tiltCurrentAngle плавно их догоняют.
  float panAngle = 90.0f;
  float tiltAngle = 90.0f;

  // Шаг для одиночной команды /api/move, если приложение использует step.
  int stepDegrees = 8;

  bool invertPan = false;
  bool invertTilt = false;
  bool laserOn = false;

  // Настройки стабильного HOME.
  // HOME работает так: сначала уезжаем в approach-точку, потом всегда заезжаем домой с одной стороны.
  float homePanAngle = 90.0f;
  float homeTiltAngle = 90.0f;

  // Pan обычно +20..+35.
  // Tilt надо подобрать: если сверху/снизу разные точки, попробуй +25, потом -25.
  float homePanApproachOffset = 25.0f;
  float homeTiltApproachOffset = 25.0f;

  // Финальный подъезд домой лучше делать медленно, чтобы выбрать люфт редуктора.
  float homeFinalSpeedDps = 70.0f;
  float homeTolerance = 0.25f;

  uint32_t homeApproachSettleMs = 500;
  uint32_t homeFinalSettleMs = 700;
} config;

// =============================================================
// GLOBAL STATE
// =============================================================
bool stationConnected = false;
bool apEnabled = false;
String apSsid;
String lastCommand = "boot";

// Функция выбора скорости, как в твоем коде раньше: slow / medium / fast.
String continuousSpeedMode = "medium";

bool servosAttached = false;
String continuousDirection = "";
unsigned long lastContinuousUpdateMs = 0;

float panCurrentAngle = 90.0f;
float tiltCurrentAngle = 90.0f;
unsigned long lastServoUpdateMs = 0;

enum HomeState {
  HOME_IDLE,
  HOME_MOVING_TO_APPROACH,
  HOME_WAIT_APPROACH,
  HOME_MOVING_TO_CENTER,
  HOME_WAIT_CENTER
};

HomeState homeState = HOME_IDLE;
unsigned long homeStateMs = 0;

// =============================================================
// UTILS
// =============================================================
float clampAngle(float value) {
  if (value < SERVO_MIN_ANGLE) return SERVO_MIN_ANGLE;
  if (value > SERVO_MAX_ANGLE) return SERVO_MAX_ANGLE;
  return value;
}

int clampPulse(int value) {
  if (value < SERVO_MIN_US) return SERVO_MIN_US;
  if (value > SERVO_MAX_US) return SERVO_MAX_US;
  return value;
}

bool parseBool(const String &value) {
  String v = value;
  v.toLowerCase();
  return v == "1" || v == "true" || v == "on" || v == "yes";
}

float limitFloat(float value, float minValue, float maxValue) {
  if (value < minValue) return minValue;
  if (value > maxValue) return maxValue;
  return value;
}

uint32_t limitU32(uint32_t value, uint32_t minValue, uint32_t maxValue) {
  if (value < minValue) return minValue;
  if (value > maxValue) return maxValue;
  return value;
}

String checkedAttr(bool value) {
  return value ? " checked" : "";
}

float getSpeedMultiplier() {
  if (continuousSpeedMode == "slow") return 0.35f;
  if (continuousSpeedMode == "medium") return 0.65f;
  return 1.0f; // fast
}

// =============================================================
// PREFERENCES
// =============================================================
void saveConfig() {
  prefs.putString("wifi_ssid", config.wifiSsid);
  prefs.putString("wifi_pass", config.wifiPassword);
  prefs.putString("host_name", config.hostName);

  prefs.putInt("pan_pin", config.panPin);
  prefs.putInt("tilt_pin", config.tiltPin);
  prefs.putInt("laser_pin", config.laserPin);

  prefs.putFloat("pan_angle", config.panAngle);
  prefs.putFloat("tilt_angle", config.tiltAngle);
  prefs.putInt("step_deg", config.stepDegrees);

  prefs.putBool("inv_pan", config.invertPan);
  prefs.putBool("inv_tilt", config.invertTilt);
  prefs.putBool("laser_on", config.laserOn);
  prefs.putString("speed_mode", continuousSpeedMode);

  prefs.putFloat("home_pan", config.homePanAngle);
  prefs.putFloat("home_tilt", config.homeTiltAngle);
  prefs.putFloat("home_pan_off", config.homePanApproachOffset);
  prefs.putFloat("home_tilt_off", config.homeTiltApproachOffset);
  prefs.putFloat("home_final_spd", config.homeFinalSpeedDps);
  prefs.putFloat("home_tol", config.homeTolerance);
  prefs.putULong("home_app_settle", config.homeApproachSettleMs);
  prefs.putULong("home_fin_settle", config.homeFinalSettleMs);
}

void loadConfig() {
  prefs.begin("pantilt", false);

  config.wifiSsid = prefs.getString("wifi_ssid", "Fenix");
  config.wifiPassword = prefs.getString("wifi_pass", "16031991");
  config.hostName = prefs.getString("host_name", "trackingcat-pantilt");

  config.panPin = prefs.getInt("pan_pin", 17);
  config.tiltPin = prefs.getInt("tilt_pin", 18);
  config.laserPin = prefs.getInt("laser_pin", 5);

  // Совместимость со старой версией, где углы хранились как int.
  float panF = prefs.getFloat("pan_angle", NAN);
  float tiltF = prefs.getFloat("tilt_angle", NAN);
  if (isnan(panF)) panF = (float)prefs.getInt("pan_angle", 90);
  if (isnan(tiltF)) tiltF = (float)prefs.getInt("tilt_angle", 90);

  config.panAngle = clampAngle(panF);
  config.tiltAngle = clampAngle(tiltF);

  config.stepDegrees = prefs.getInt("step_deg", 8);
  if (config.stepDegrees <= 0) config.stepDegrees = 8;
  if (config.stepDegrees > 45) config.stepDegrees = 45;

  config.invertPan = prefs.getBool("inv_pan", false);
  config.invertTilt = prefs.getBool("inv_tilt", false);
  config.laserOn = prefs.getBool("laser_on", false);

  continuousSpeedMode = prefs.getString("speed_mode", "medium");
  if (continuousSpeedMode != "slow" && continuousSpeedMode != "medium" && continuousSpeedMode != "fast") {
    continuousSpeedMode = "medium";
  }

  config.homePanAngle = clampAngle(prefs.getFloat("home_pan", 90.0f));
  config.homeTiltAngle = clampAngle(prefs.getFloat("home_tilt", 90.0f));

  config.homePanApproachOffset = limitFloat(prefs.getFloat("home_pan_off", 25.0f), -60.0f, 60.0f);
  config.homeTiltApproachOffset = limitFloat(prefs.getFloat("home_tilt_off", 25.0f), -60.0f, 60.0f);

  config.homeFinalSpeedDps = limitFloat(prefs.getFloat("home_final_spd", 70.0f), 20.0f, 250.0f);
  config.homeTolerance = limitFloat(prefs.getFloat("home_tol", 0.25f), 0.05f, 3.0f);

  config.homeApproachSettleMs = limitU32(prefs.getULong("home_app_settle", 500), 50, 3000);
  config.homeFinalSettleMs = limitU32(prefs.getULong("home_fin_settle", 700), 50, 3000);
}

// =============================================================
// SERVO CONTROL
// =============================================================
void ensureServosAttached() {
  if (servosAttached) return;

  panServo.setPeriodHertz(50);
  tiltServo.setPeriodHertz(50);

  panServo.attach(config.panPin, SERVO_MIN_US, SERVO_MAX_US);
  tiltServo.attach(config.tiltPin, SERVO_MIN_US, SERVO_MAX_US);

  servosAttached = true;
}

int angleToPulse(float angle) {
  angle = clampAngle(angle);
  float t = (angle - SERVO_MIN_ANGLE) / (SERVO_MAX_ANGLE - SERVO_MIN_ANGLE);
  int us = (int)lroundf(SERVO_MIN_US + t * (SERVO_MAX_US - SERVO_MIN_US));
  return clampPulse(us);
}

void writeServoRaw(float panAngle, float tiltAngle) {
  ensureServosAttached();

  float panLogical = clampAngle(panAngle);
  float tiltLogical = clampAngle(tiltAngle);

  float panOut = config.invertPan ? 180.0f - panLogical : panLogical;
  float tiltOut = config.invertTilt ? 180.0f - tiltLogical : tiltLogical;

  // writeMicroseconds дает более мелкое разрешение, чем Servo.write(angle).
  panServo.writeMicroseconds(angleToPulse(panOut));
  tiltServo.writeMicroseconds(angleToPulse(tiltOut));
}

float moveTowards(float current, float target, float maxDelta) {
  float delta = target - current;

  if (fabsf(delta) <= maxDelta) {
    return target;
  }

  return current + (delta > 0.0f ? maxDelta : -maxDelta);
}

void setTargetAngles(float pan, float tilt) {
  config.panAngle = clampAngle(pan);
  config.tiltAngle = clampAngle(tilt);
}

bool isNearCommandedTarget(float panTarget, float tiltTarget) {
  return fabsf(panCurrentAngle - panTarget) <= config.homeTolerance &&
         fabsf(tiltCurrentAngle - tiltTarget) <= config.homeTolerance;
}

void cancelHomeRoutine() {
  homeState = HOME_IDLE;
}

void startHomeRoutine() {
  continuousDirection = "";
  lastContinuousUpdateMs = 0;

  float approachPan = clampAngle(config.homePanAngle + config.homePanApproachOffset);
  float approachTilt = clampAngle(config.homeTiltAngle + config.homeTiltApproachOffset);

  // Сначала уходим в approach-точку, затем финальный подъезд домой всегда будет с одной стороны.
  setTargetAngles(approachPan, approachTilt);

  homeState = HOME_MOVING_TO_APPROACH;
  homeStateMs = millis();
}

void updateContinuousTarget() {
  if (continuousDirection.length() == 0) {
    lastContinuousUpdateMs = 0;
    return;
  }

  unsigned long now = millis();
  if (lastContinuousUpdateMs == 0) {
    lastContinuousUpdateMs = now;
    return;
  }

  float dt = (now - lastContinuousUpdateMs) / 1000.0f;
  lastContinuousUpdateMs = now;

  if (dt <= 0.0f || dt > 0.2f) return;

  float speedDps = CONTINUOUS_TARGET_SPEED_DPS * getSpeedMultiplier();
  float delta = speedDps * dt;

  if (continuousDirection == "left") {
    config.panAngle = clampAngle(config.panAngle - delta);
  } else if (continuousDirection == "right") {
    config.panAngle = clampAngle(config.panAngle + delta);
  } else if (continuousDirection == "up") {
    config.tiltAngle = clampAngle(config.tiltAngle - delta);
  } else if (continuousDirection == "down") {
    config.tiltAngle = clampAngle(config.tiltAngle + delta);
  }
}

void updateSmoothServos() {
  unsigned long now = millis();

  if (now - lastServoUpdateMs < SERVO_UPDATE_MS) return;

  float dt;
  if (lastServoUpdateMs == 0) {
    dt = SERVO_UPDATE_MS / 1000.0f;
  } else {
    dt = (now - lastServoUpdateMs) / 1000.0f;
  }

  lastServoUpdateMs = now;

  if (dt <= 0.0f || dt > 0.2f) return;

  float speedDps = SERVO_SPEED_DPS;

  // Финальный подъезд HOME делаем медленнее, чтобы редуктор выбрал люфт одинаково.
  if (homeState == HOME_MOVING_TO_CENTER) {
    speedDps = config.homeFinalSpeedDps;
  }

  float maxDelta = speedDps * dt;

  float targetPan = clampAngle(config.panAngle);
  float targetTilt = clampAngle(config.tiltAngle);

  panCurrentAngle = moveTowards(panCurrentAngle, targetPan, maxDelta);
  tiltCurrentAngle = moveTowards(tiltCurrentAngle, targetTilt, maxDelta);

  writeServoRaw(panCurrentAngle, tiltCurrentAngle);
}

void updateHomeRoutine() {
  if (homeState == HOME_IDLE) return;

  unsigned long now = millis();

  float approachPan = clampAngle(config.homePanAngle + config.homePanApproachOffset);
  float approachTilt = clampAngle(config.homeTiltAngle + config.homeTiltApproachOffset);

  if (homeState == HOME_MOVING_TO_APPROACH) {
    if (isNearCommandedTarget(approachPan, approachTilt)) {
      homeState = HOME_WAIT_APPROACH;
      homeStateMs = now;
    }
    return;
  }

  if (homeState == HOME_WAIT_APPROACH) {
    if (now - homeStateMs >= config.homeApproachSettleMs) {
      setTargetAngles(config.homePanAngle, config.homeTiltAngle);
      homeState = HOME_MOVING_TO_CENTER;
      homeStateMs = now;
    }
    return;
  }

  if (homeState == HOME_MOVING_TO_CENTER) {
    if (isNearCommandedTarget(config.homePanAngle, config.homeTiltAngle)) {
      homeState = HOME_WAIT_CENTER;
      homeStateMs = now;
    }
    return;
  }

  if (homeState == HOME_WAIT_CENTER) {
    if (now - homeStateMs >= config.homeFinalSettleMs) {
      // Фиксируем логическое состояние точно в HOME и продолжаем удерживать PWM.
      config.panAngle = config.homePanAngle;
      config.tiltAngle = config.homeTiltAngle;
      panCurrentAngle = config.homePanAngle;
      tiltCurrentAngle = config.homeTiltAngle;
      writeServoRaw(panCurrentAngle, tiltCurrentAngle);
      homeState = HOME_IDLE;
      lastCommand = "home_done";
    }
  }
}

void attachHardware() {
  pinMode(config.laserPin, OUTPUT);

  ensureServosAttached();

  config.panAngle = clampAngle(config.panAngle);
  config.tiltAngle = clampAngle(config.tiltAngle);

  panCurrentAngle = config.panAngle;
  tiltCurrentAngle = config.tiltAngle;

  writeServoRaw(panCurrentAngle, tiltCurrentAngle);
  digitalWrite(config.laserPin, config.laserOn ? HIGH : LOW);
}

// =============================================================
// WIFI
// =============================================================
void startAccessPoint() {
  apSsid = "TrackingCatPanTilt-" + String((uint32_t)(ESP.getEfuseMac() & 0xFFFFFF), HEX);

  WiFi.mode(WIFI_AP_STA);
  WiFi.softAP(apSsid.c_str(), "trackingcat");

  apEnabled = true;
}

void connectWifi() {
  stationConnected = false;
  apEnabled = false;

  if (config.wifiSsid.isEmpty()) {
    startAccessPoint();
    return;
  }

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false); // меньше задержек Wi-Fi
  WiFi.setHostname(config.hostName.c_str());
  WiFi.begin(config.wifiSsid.c_str(), config.wifiPassword.c_str());

  unsigned long started = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - started < 15000) {
    delay(250);
  }

  if (WiFi.status() == WL_CONNECTED) {
    stationConnected = true;

    if (MDNS.begin(config.hostName.c_str())) {
      MDNS.addService("http", "tcp", 80);
    }
  } else {
    startAccessPoint();
  }
}

String currentIp() {
  if (stationConnected) return WiFi.localIP().toString();
  if (apEnabled) return WiFi.softAPIP().toString();
  return "0.0.0.0";
}

String wifiMode() {
  if (stationConnected && apEnabled) return "sta+ap";
  if (stationConnected) return "sta";
  if (apEnabled) return "ap";
  return "offline";
}

// =============================================================
// JSON / HTTP HELPERS
// =============================================================
void sendCorsHeaders() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.sendHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  server.sendHeader("Access-Control-Allow-Headers", "Content-Type");
}

void sendStateJson(const char* status = "ok") {
  StaticJsonDocument<1152> doc;

  doc["status"] = status;
  doc["connected"] = true;
  doc["wifi_mode"] = wifiMode();
  doc["ip_address"] = currentIp();
  doc["host_name"] = config.hostName;

  doc["pan_pin"] = config.panPin;
  doc["tilt_pin"] = config.tiltPin;
  doc["laser_pin"] = config.laserPin;

  doc["pan_angle"] = config.panAngle;
  doc["tilt_angle"] = config.tiltAngle;
  doc["pan_current_angle"] = panCurrentAngle;
  doc["tilt_current_angle"] = tiltCurrentAngle;
  doc["step_degrees"] = config.stepDegrees;

  doc["speed_mode"] = continuousSpeedMode;
  doc["speed_multiplier"] = getSpeedMultiplier();

  doc["invert_pan"] = config.invertPan;
  doc["invert_tilt"] = config.invertTilt;
  doc["laser_on"] = config.laserOn;

  doc["servo_speed_dps"] = SERVO_SPEED_DPS;
  doc["continuous_target_speed_dps"] = CONTINUOUS_TARGET_SPEED_DPS;
  doc["servo_update_ms"] = SERVO_UPDATE_MS;

  doc["home_state"] = (int)homeState;
  doc["home_pan_angle"] = config.homePanAngle;
  doc["home_tilt_angle"] = config.homeTiltAngle;
  doc["home_pan_approach_offset"] = config.homePanApproachOffset;
  doc["home_tilt_approach_offset"] = config.homeTiltApproachOffset;
  doc["home_final_speed_dps"] = config.homeFinalSpeedDps;
  doc["home_tolerance"] = config.homeTolerance;
  doc["home_approach_settle_ms"] = config.homeApproachSettleMs;
  doc["home_final_settle_ms"] = config.homeFinalSettleMs;

  doc["last_command"] = lastCommand;
  doc["ap_ssid"] = apEnabled ? apSsid : "";
  doc["station_ssid"] = config.wifiSsid;

  String payload;
  serializeJson(doc, payload);

  sendCorsHeaders();
  server.send(200, "application/json", payload);
}

// =============================================================
// WEB UI
// =============================================================
void handleRoot() {
  String html;

  html += "<!doctype html><html><head><meta charset='utf-8'>";
  html += "<meta name='viewport' content='width=device-width,initial-scale=1'>";
  html += "<title>TrackingCat PanTilt</title>";
  html += "<style>";
  html += "body{font-family:Arial,sans-serif;background:#111;color:#eee;padding:18px}";
  html += "button,input{font-size:16px;padding:10px;margin:6px;border-radius:8px;border:1px solid #444}";
  html += "button{background:#2b2b2b;color:#fff;cursor:pointer}";
  html += ".grid{display:grid;grid-template-columns:80px 80px 80px;gap:10px;max-width:280px}";
  html += ".wide{min-width:120px}";
  html += ".card{background:#1b1b1b;border:1px solid #333;border-radius:12px;padding:16px;margin-bottom:16px}";
  html += "code{background:#222;padding:2px 5px;border-radius:4px}";
  html += "a{color:#8ab4ff}";
  html += ".active{border-color:#7aa2ff;background:#1d3766}";
  html += "</style></head><body>";

  html += "<h2>TrackingCat PanTilt</h2>";

  html += "<div class='card'>";
  html += "<p><b>IP:</b> " + currentIp() + " | <b>Mode:</b> " + wifiMode() + "</p>";
  html += "<p><b>Pan target:</b> " + String(config.panAngle, 1) + " | <b>Tilt target:</b> " + String(config.tiltAngle, 1) + "</p>";
  html += "<p><b>Speed:</b> " + continuousSpeedMode + " | <b>Home state:</b> " + String((int)homeState) + "</p>";
  html += "<p><a href='/api/state'>JSON state</a></p>";

  html += "<div class='grid'>";
  html += "<div></div><button onclick=\"fetch('/api/move?tilt_delta=-15').then(()=>location.reload())\">UP</button><div></div>";
  html += "<button onclick=\"fetch('/api/move?pan_delta=-15').then(()=>location.reload())\">LEFT</button>";
  html += "<button onclick=\"fetch('/api/center').then(()=>location.reload())\">HOME</button>";
  html += "<button onclick=\"fetch('/api/move?pan_delta=15').then(()=>location.reload())\">RIGHT</button>";
  html += "<div></div><button onclick=\"fetch('/api/move?tilt_delta=15').then(()=>location.reload())\">DOWN</button><div></div>";
  html += "</div>";

  html += "<p>";
  html += "<button class='wide' onclick=\"fetch('/api/laser/toggle').then(()=>location.reload())\">Toggle laser</button>";
  html += "<button class='wide' onclick=\"fetch('/api/save_position').then(()=>location.reload())\">Save position</button>";
  html += "</p>";
  html += "</div>";

  html += "<div class='card'>";
  html += "<h3>Speed mode</h3>";
  html += "<button class='" + String(continuousSpeedMode == "slow" ? "active" : "") + "' onclick=\"fetch('/api/speed?mode=slow').then(()=>location.reload())\">Slow</button>";
  html += "<button class='" + String(continuousSpeedMode == "medium" ? "active" : "") + "' onclick=\"fetch('/api/speed?mode=medium').then(()=>location.reload())\">Medium</button>";
  html += "<button class='" + String(continuousSpeedMode == "fast" ? "active" : "") + "' onclick=\"fetch('/api/speed?mode=fast').then(()=>location.reload())\">Fast</button>";
  html += "<p>API: <code>/api/speed?mode=slow</code>, <code>medium</code>, <code>fast</code></p>";
  html += "</div>";

  html += "<div class='card'>";
  html += "<h3>Stable HOME setup</h3>";
  html += "<form action='/api/home_config'>";
  html += "<input name='home_pan' placeholder='Home pan' value='" + String(config.homePanAngle, 1) + "'>";
  html += "<input name='home_tilt' placeholder='Home tilt' value='" + String(config.homeTiltAngle, 1) + "'><br>";
  html += "<input name='pan_offset' placeholder='Pan offset' value='" + String(config.homePanApproachOffset, 1) + "'>";
  html += "<input name='tilt_offset' placeholder='Tilt offset' value='" + String(config.homeTiltApproachOffset, 1) + "'><br>";
  html += "<input name='final_speed' placeholder='Final speed dps' value='" + String(config.homeFinalSpeedDps, 1) + "'>";
  html += "<input name='tolerance' placeholder='Tolerance' value='" + String(config.homeTolerance, 2) + "'><br>";
  html += "<input name='approach_settle' placeholder='Approach settle ms' value='" + String(config.homeApproachSettleMs) + "'>";
  html += "<input name='final_settle' placeholder='Final settle ms' value='" + String(config.homeFinalSettleMs) + "'>";
  html += "<button type='submit'>Save HOME</button>";
  html += "</form>";
  html += "<p>Если HOME сверху и снизу отличается, попробуй <code>tilt_offset=25</code>, потом <code>tilt_offset=-25</code>. Лучший вариант оставь.</p>";
  html += "</div>";

  html += "<div class='card'>";
  html += "<h3>Wi-Fi setup</h3>";
  html += "<form action='/api/config/wifi'>";
  html += "<input name='ssid' placeholder='SSID' value='" + config.wifiSsid + "'>";
  html += "<input name='password' placeholder='Password' value='" + config.wifiPassword + "'>";
  html += "<input name='host_name' placeholder='Host name' value='" + config.hostName + "'>";
  html += "<button type='submit'>Save Wi-Fi</button>";
  html += "</form>";
  html += "<p>If Wi-Fi is not configured, the board starts AP mode. Connect to the ESP32 AP and open <code>192.168.4.1</code>.</p>";
  html += "</div>";

  html += "<div class='card'>";
  html += "<h3>Pin setup</h3>";
  html += "<form action='/api/config/pins'>";
  html += "<input name='pan_pin' placeholder='Pan GPIO' value='" + String(config.panPin) + "'>";
  html += "<input name='tilt_pin' placeholder='Tilt GPIO' value='" + String(config.tiltPin) + "'>";
  html += "<input name='laser_pin' placeholder='Laser GPIO' value='" + String(config.laserPin) + "'>";
  html += "<button type='submit'>Save pins</button>";
  html += "</form>";
  html += "</div>";

  html += "<div class='card'>";
  html += "<h3>Motion setup</h3>";
  html += "<form action='/api/step'>";
  html += "<input name='degrees' placeholder='Step degrees' value='" + String(config.stepDegrees) + "'>";
  html += "<button type='submit'>Save step</button>";
  html += "</form>";
  html += "<form action='/api/config/invert'>";
  html += "<label><input type='checkbox' name='invert_pan' value='1'" + checkedAttr(config.invertPan) + "> Invert pan</label><br>";
  html += "<label><input type='checkbox' name='invert_tilt' value='1'" + checkedAttr(config.invertTilt) + "> Invert tilt</label><br>";
  html += "<button type='submit'>Save inversion</button>";
  html += "</form>";
  html += "</div>";

  html += "<div class='card'>";
  html += "<h3>API</h3>";
  html += "<p><code>/api/move?pan_delta=15</code></p>";
  html += "<p><code>/api/move?tilt_delta=-15</code></p>";
  html += "<p><code>/api/start_move?direction=left</code>, <code>right</code>, <code>up</code>, <code>down</code></p>";
  html += "<p><code>/api/stop</code></p>";
  html += "<p><code>/api/target?pan=90&tilt=90</code></p>";
  html += "<p><code>/api/center</code></p>";
  html += "<p><code>/api/home_config?tilt_offset=-25</code></p>";
  html += "</div>";

  html += "</body></html>";

  server.send(200, "text/html; charset=utf-8", html);
}

// =============================================================
// API HANDLERS
// =============================================================
void handleMove() {
  cancelHomeRoutine();
  continuousDirection = "";

  float panDelta = server.hasArg("pan_delta") ? server.arg("pan_delta").toFloat() : 0.0f;
  float tiltDelta = server.hasArg("tilt_delta") ? server.arg("tilt_delta").toFloat() : 0.0f;

  setTargetAngles(config.panAngle + panDelta, config.tiltAngle + tiltDelta);

  lastCommand = "move";
  sendStateJson();
}

void handleTarget() {
  cancelHomeRoutine();
  continuousDirection = "";

  float pan = server.hasArg("pan") ? server.arg("pan").toFloat() : config.panAngle;
  float tilt = server.hasArg("tilt") ? server.arg("tilt").toFloat() : config.tiltAngle;

  setTargetAngles(pan, tilt);

  lastCommand = "target";
  sendStateJson();
}

void handleStartMove() {
  cancelHomeRoutine();

  if (!server.hasArg("direction")) {
    sendCorsHeaders();
    server.send(400, "application/json", "{\"error\":\"missing direction\"}");
    return;
  }

  String direction = server.arg("direction");

  if (direction != "left" && direction != "right" && direction != "up" && direction != "down") {
    sendCorsHeaders();
    server.send(400, "application/json", "{\"error\":\"bad direction\"}");
    return;
  }

  continuousDirection = direction;
  lastContinuousUpdateMs = 0;

  lastCommand = "start_move";
  sendStateJson();
}

void handleCenter() {
  startHomeRoutine();

  lastCommand = "home_started";
  sendStateJson("homing_started");
}

void handleLaser(bool toggleOnly) {
  if (toggleOnly) {
    config.laserOn = !config.laserOn;
  } else {
    config.laserOn = server.hasArg("on") ? parseBool(server.arg("on")) : config.laserOn;
  }

  digitalWrite(config.laserPin, config.laserOn ? HIGH : LOW);

  saveConfig();
  lastCommand = "laser";
  sendStateJson();
}

void handleSpeed() {
  String mode = server.hasArg("mode") ? server.arg("mode") : continuousSpeedMode;
  mode.toLowerCase();

  if (mode != "slow" && mode != "medium" && mode != "fast") {
    sendCorsHeaders();
    server.send(400, "application/json", "{\"error\":\"invalid speed mode\"}");
    return;
  }

  continuousSpeedMode = mode;
  saveConfig();
  lastCommand = "speed";
  sendStateJson();
}

void handleStep() {
  int degrees = server.hasArg("degrees") ? server.arg("degrees").toInt() : config.stepDegrees;

  if (degrees <= 0) degrees = 1;
  if (degrees > 45) degrees = 45;

  config.stepDegrees = degrees;

  saveConfig();
  lastCommand = "step";
  sendStateJson();
}

void handleStop() {
  continuousDirection = "";
  lastContinuousUpdateMs = 0;

  lastCommand = "stop";
  sendStateJson();
}

void handleWifiConfig() {
  if (server.hasArg("ssid")) config.wifiSsid = server.arg("ssid");
  if (server.hasArg("password")) config.wifiPassword = server.arg("password");
  if (server.hasArg("host_name") && server.arg("host_name").length() > 0) {
    config.hostName = server.arg("host_name");
  }

  saveConfig();
  lastCommand = "wifi-config";
  sendStateJson("saved_reboot_required");
}

void handlePinConfig() {
  if (server.hasArg("pan_pin")) config.panPin = server.arg("pan_pin").toInt();
  if (server.hasArg("tilt_pin")) config.tiltPin = server.arg("tilt_pin").toInt();
  if (server.hasArg("laser_pin")) config.laserPin = server.arg("laser_pin").toInt();

  saveConfig();
  lastCommand = "pin-config";
  sendStateJson("saved_reboot_required");
}

void handleInvertConfig() {
  config.invertPan = server.hasArg("invert_pan") && parseBool(server.arg("invert_pan"));
  config.invertTilt = server.hasArg("invert_tilt") && parseBool(server.arg("invert_tilt"));

  writeServoRaw(panCurrentAngle, tiltCurrentAngle);

  saveConfig();
  lastCommand = "invert-config";
  sendStateJson();
}

void handleHomeConfig() {
  cancelHomeRoutine();

  if (server.hasArg("home_pan")) {
    config.homePanAngle = clampAngle(server.arg("home_pan").toFloat());
  }
  if (server.hasArg("home_tilt")) {
    config.homeTiltAngle = clampAngle(server.arg("home_tilt").toFloat());
  }
  if (server.hasArg("pan_offset")) {
    config.homePanApproachOffset = limitFloat(server.arg("pan_offset").toFloat(), -60.0f, 60.0f);
  }
  if (server.hasArg("tilt_offset")) {
    config.homeTiltApproachOffset = limitFloat(server.arg("tilt_offset").toFloat(), -60.0f, 60.0f);
  }
  if (server.hasArg("final_speed")) {
    config.homeFinalSpeedDps = limitFloat(server.arg("final_speed").toFloat(), 20.0f, 250.0f);
  }
  if (server.hasArg("tolerance")) {
    config.homeTolerance = limitFloat(server.arg("tolerance").toFloat(), 0.05f, 3.0f);
  }
  if (server.hasArg("approach_settle")) {
    config.homeApproachSettleMs = limitU32((uint32_t)server.arg("approach_settle").toInt(), 50, 3000);
  }
  if (server.hasArg("final_settle")) {
    config.homeFinalSettleMs = limitU32((uint32_t)server.arg("final_settle").toInt(), 50, 3000);
  }

  saveConfig();
  lastCommand = "home-config";
  sendStateJson();
}

void handleSavePosition() {
  config.panAngle = clampAngle(config.panAngle);
  config.tiltAngle = clampAngle(config.tiltAngle);

  saveConfig();
  lastCommand = "save-position";
  sendStateJson("position_saved");
}

void handleReboot() {
  sendStateJson("rebooting");
  delay(200);
  ESP.restart();
}

void handleOptions() {
  sendCorsHeaders();
  server.send(204);
}

// =============================================================
// SETUP / LOOP
// =============================================================
void setup() {
  Serial.begin(115200);
  delay(200);

  loadConfig();
  attachHardware();
  connectWifi();

  server.on("/", HTTP_GET, handleRoot);

  server.on("/api/state", HTTP_GET, []() {
    sendStateJson();
  });

  server.on("/api/move", HTTP_GET, handleMove);
  server.on("/api/target", HTTP_GET, handleTarget);
  server.on("/api/start_move", HTTP_GET, handleStartMove);
  server.on("/api/center", HTTP_GET, handleCenter);

  server.on("/api/laser", HTTP_GET, []() {
    handleLaser(false);
  });

  server.on("/api/laser/toggle", HTTP_GET, []() {
    handleLaser(true);
  });

  server.on("/api/step", HTTP_GET, handleStep);
  server.on("/api/speed", HTTP_GET, handleSpeed);
  server.on("/api/stop", HTTP_GET, handleStop);
  server.on("/api/save_position", HTTP_GET, handleSavePosition);

  server.on("/api/config/wifi", HTTP_GET, handleWifiConfig);
  server.on("/api/config/pins", HTTP_GET, handlePinConfig);
  server.on("/api/config/invert", HTTP_GET, handleInvertConfig);
  server.on("/api/home_config", HTTP_GET, handleHomeConfig);
  server.on("/api/reboot", HTTP_GET, handleReboot);

  server.onNotFound([]() {
    if (server.method() == HTTP_OPTIONS) {
      handleOptions();
    } else {
      sendCorsHeaders();
      server.send(404, "application/json", "{\"error\":\"not found\"}");
    }
  });

  server.begin();

  lastCommand = "ready";
}

void loop() {
  server.handleClient();

  updateContinuousTarget();
  updateSmoothServos();
  updateHomeRoutine();

  // Серво НЕ отключаем. Detach для pan-tilt дает рывки и потерю удержания.
  delay(1);
}
