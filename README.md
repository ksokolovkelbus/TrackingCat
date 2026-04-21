# TrackingCat

`TrackingCat` определяет кошек через YOLO, ведет стабильные `track_id` в multi-cat detect-then-track режиме и поверх этого добавляет:

- ручные `scene zones`;
- интерактивный `zone editor` в OpenCV окне;
- звуковой alert при входе кошки в опасную зону;
- непрерывный alarm для выбранных типов зон;
- автоматическую запись коротких incident-видео с prebuffer/postbuffer;
- хранение зон в `normalized` координатах, чтобы они не съезжали при другом фактическом разрешении кадра.

Важно: detect-then-track ядро не переписывалось. Зоны, звук и запись инцидентов работают как надстройка над уже готовыми `visible_tracks`.

## Что Изменилось В Alert Логике

Главный источник ложных срабатываний раньше был такой:

- bbox кошки был большим;
- край bbox пересекал `restricted` или `surface`;
- alert срабатывал раньше, чем кошка реально заходила в зону.

Теперь alert считается не по bbox edge и не по overlap, а только по одной точке:

- `surface_alert.alert_point_mode: crosshair_center`

Это ровно та же точка, по которой на экране рисуется крестик. Если крестик не вошел в зону, тревоги не будет, даже если bbox уже частично наехал на нее.

`bbox_overlap_threshold` и старая геометрия зоны в коде сохранены как вспомогательная функциональность, но для сигнала и incident recording они больше не используются.

## Что Сейчас Включено По Умолчанию

Текущее `configs/default.yaml` уже содержит ваши реальные зоны:

- `restricted_1`
- `restricted_2`
- `floor_1`

Также по умолчанию включено:

- `overlay.show_track_boxes: false`
- `overlay.show_track_crosshair: true`
- `surface_alert.alert_point_mode: crosshair_center`
- `surface_alert.sound_file: sounds/cat_alert.wav`
- `surface_alert.continuous_while_in_zone: true`
- `surface_alert.continuous_zone_types: [restricted]`
- `alert_recording.enabled: true`
- `alert_recording.output_dir: /home/kelbus/PycharmProjects/TrackingCat/alert_video`

То есть текущий runtime по умолчанию:

- большие track-boxes не рисуются;
- крестик и подпись кошки остаются;
- разовый alert идет только когда сам крестик зашел в `surface` или `restricted`;
- если кошка остается в `restricted`, включается continuous audio;
- при активной тревоге автоматически создается incident-видео.

## Архитектура

Порядок работы остался таким:

1. `frame`
2. `detector / tracker update`
3. готовые `active tracks`
4. point-based zone classification для alert
5. `surface / restricted event generation`
6. `audio alert / continuous alarm`
7. render обычного overlay
8. incident recording поверх уже отрисованного кадра

Это означает:

- multi-cat tracking не сломан;
- detect-then-track scheduler не менялся;
- zone editor и YAML workflow сохранены;
- continuous audio alert остался и теперь работает с тем же point-based zone state.

## Где Что Настраивать

- основной конфиг: `configs/default.yaml`
- multi-cat tracking: секция `tracking`
- зоны сцены: секция `scene_zones`
- разовый и continuous alert: секция `surface_alert`
- incident recording: секция `alert_recording`
- модели и dataclass-конфиг: `app/models.py`
- загрузка и валидация YAML: `app/config.py`
- геометрия зон и alert-point classification: `app/zones.py`
- монитор зон и alert state: `app/surface_monitor.py`
- incident recorder: `app/alert_recorder.py`
- overlay: `app/overlay.py`
- runtime pipeline: `app/main.py`
- OpenCV zone editor: `app/zone_editor.py`

## Запуск

Обычный запуск:

```bash
python -m app.main --config configs/default.yaml
```

или:

```bash
.venv/bin/python -m app.main --config configs/default.yaml
```

Выход из обычного runtime:

- `q`

## Запуск Zone Editor

Через CLI:

```bash
python -m app.main --config configs/default.yaml --zone-editor true
```

или:

```bash
.venv/bin/python -m app.main --config configs/default.yaml --zone-editor true
```

Можно включить editor и из YAML:

```yaml
scene_zones:
  zone_editor_enabled: true
```

Но обычно удобнее держать это выключенным и запускать editor отдельным флагом.

## Управление В Zone Editor

Мышь:

- левая кнопка:
  - в режиме `rect` начинает и завершает прямоугольник;
  - в режиме `polygon` добавляет точку;
  - если draft пустой и клик попал в существующую зону, зона выбирается;
- правая кнопка:
  - завершает полигон.

Клавиши:

- `r` — режим прямоугольника
- `p` — режим полигона
- `f` — тип зоны `floor`
- `s` — тип зоны `surface`
- `x` — тип зоны `restricted`
- `u` — undo: удалить последнюю точку draft или последнюю/выбранную зону
- `c` — очистить текущий draft
- `d` — удалить выбранную зону
- `n` — ввести имя зоны
- `w` — сохранить зоны в YAML
- `l` — перечитать зоны из YAML
- `q` — выйти

Во время ввода имени:

- `Enter` — подтвердить
- `Backspace` — удалить символ
- `Esc` — отменить ввод

Отдельного режима `line` сейчас нет. Если нужна граница пола, рисуй ее полигоном.

## Как Сохраняются Зоны

Редактор сохраняет только секцию `scene_zones` в том YAML, который передан через `--config`.

Это дает две вещи:

- остальные секции конфига не перетираются;
- runtime сразу использует те же зоны без ручного копирования координат.

После сохранения зоны всегда записываются как `normalized`, даже если раньше они были в пикселях.

## Формат Зон В YAML

Поддерживаются:

- `zone_type: floor`
- `zone_type: surface`
- `zone_type: restricted`
- `shape_type: rect`
- `shape_type: polygon`
- `coordinates_mode: normalized`
- `coordinates_mode: pixels`

Пример:

```yaml
scene_zones:
  enabled: true
  draw_zones: true
  draw_track_locations: true
  surface_priority_over_floor: true
  bbox_overlap_threshold: 0.15
  coordinates_mode: normalized
  zone_editor_enabled: false
  zones:
    - name: floor_1
      enabled: true
      zone_type: floor
      shape_type: polygon
      coordinates_mode: normalized
      points:
        - [0.01, 0.60]
        - [0.43, 0.99]
        - [0.99, 0.99]
        - [0.99, 0.34]

    - name: restricted_1
      enabled: true
      zone_type: restricted
      shape_type: polygon
      coordinates_mode: normalized
      points:
        - [0.20, 0.30]
        - [0.36, 0.35]
        - [0.69, 0.27]
        - [0.99, 0.33]
```

## Почему Normalized Coordinates Лучше Pixels

Старая проблема была такой:

- в конфиге просили, например, `960x540`;
- камера реально открывалась как `640x480`;
- зоны оставались в старых пикселях;
- overlay и classification съезжали.

Теперь зона хранится как доля от ширины и высоты кадра:

- `x = 0.25`
- `y = 0.60`

На каждом кадре координаты проецируются в текущий `frame.shape`, поэтому:

- зоны не зависят от реального разрешения камеры;
- editor можно использовать на любом текущем размере кадра;
- один и тот же YAML работает стабильнее.

Старый pixel format сохраняет backward compatibility:

- если координаты больше `1.0`, зона читается как `pixels`;
- при следующем сохранении editor перепишет ее в `normalized`.

## Как Теперь Определяется Зона Для Alert

Для сигнала и записи используется только crosshair-center:

- для каждого `track_id` берется `track.center`;
- эта точка проверяется через `point_in_zone(...)`;
- если точка внутри `restricted` или `surface`, зона считается активной;
- если точка снаружи, alert нет.

То есть:

- bbox может быть большим;
- bbox может частично наехать на зону;
- но если крестик еще не вошел, тревоги и записи не будет.

Это ровно то поведение, которое видно глазами на экране.

## Два Режима Звукового Сигнала

Сейчас в проекте есть два независимых механизма.

### 1. Разовый Entry Alert

Срабатывает при входе креста-кроссхейра в `surface-like` зону.

На него влияют:

- `trigger_on_surface_entry`
- `alert_point_mode`
- `trigger_only_from_floor`
- `trigger_from_unknown`
- `cooldown_seconds`
- `min_interval_per_track`
- `global_min_interval`
- `repeat_on_same_surface`

Если нужен строгий режим только `floor -> surface`, используй:

```yaml
surface_alert:
  trigger_only_from_floor: true
  trigger_from_unknown: false
```

### 2. Continuous Alarm

Это отдельный режим поверх обычного entry alert.

Если включено:

```yaml
surface_alert:
  continuous_while_in_zone: true
  continuous_zone_types:
    - restricted
  repeat_interval_seconds: 1.0
  stop_when_zone_empty: true
```

то звук будет повторяться, пока хотя бы один alert-active кот остается в одной из указанных зон.

Текущее default-поведение:

- continuous alarm включается только для `restricted`;
- повторяет звук примерно раз в `repeat_interval_seconds`;
- останавливается, когда в этих зонах никого не осталось;
- останавливается и при shutdown приложения.

## Как Играет Звук

Порядок такой:

1. Если задан `sound_file`, сначала пробуем проиграть его.
2. Если проиграть файл не удалось, идем в fallback.
3. Fallback:
   - Windows: `winsound`
   - macOS: `afplay` или системный звук
   - Linux: `paplay`, `aplay`, `ffplay`, `play`
   - если ничего нет: `terminal bell`

Приложение не падает, если backend недоступен. Ошибки только логируются.

Чтобы отключить звук:

```yaml
surface_alert:
  enabled: false
```

Чтобы оставить только fallback beep:

```yaml
surface_alert:
  sound_file: null
  beep_fallback: true
```

## Incident Recording

Теперь проект автоматически сохраняет короткий ролик инцидента в момент тревоги.

Логика session-based:

1. Пока нет alert-active котов:
   - recorder в idle;
   - хранит только кольцевой `prebuffer`.
2. Как только первый alert-active кот входит в зону:
   - стартует новый incident;
   - создается writer;
   - сначала в файл пишется `prebuffer`;
   - затем текущий кадр и все последующие кадры инцидента.
3. Пока хотя бы один alert-active кот остается в зоне:
   - запись продолжается;
   - состояние обновляется по активным track ids и zone names.
4. Когда все вышли:
   - пишется `postbuffer`;
   - writer закрывается;
   - файл переименовывается в финальное информативное имя;
   - рядом сохраняется `.json` с metadata.

Если через короткое время кто-то снова вошел в alert-зону, начинается новый incident новым файлом.

## Где Лежат Incident Videos

Все alert videos сохраняются строго сюда:

```text
/home/kelbus/PycharmProjects/TrackingCat/alert_video
```

Если директории нет, она создается автоматически.

Имена файлов формируются так:

- `2026-04-20_20-10-31_zone-restricted_1_tracks-3.mp4`
- `2026-04-20_20-10-31_zone-restricted_1_tracks-1-3.mp4`

Рядом сохраняется sidecar metadata:

- `2026-04-20_20-10-31_zone-restricted_1_tracks-1-3.json`

В `.json` записываются:

- время начала и конца incident;
- все zone names, замеченные за incident;
- все track ids / display numbers, участвовавшие в incident;
- длительность incident;
- dwell time по каждому track;
- first entered / last exited для каждого track.

## Настройки Incident Recorder

Секция:

```yaml
alert_recording:
  enabled: true
  output_dir: /home/kelbus/PycharmProjects/TrackingCat/alert_video
  prebuffer_seconds: 3.0
  postbuffer_seconds: 1.5
  fps_fallback: 15.0
  codec: mp4v
  draw_recording_overlay: true
  include_wallclock_timestamp: true
  include_zone_name: true
  include_track_ids: true
  include_elapsed_seconds: true
```

Главные ручки:

- `prebuffer_seconds` — сколько секунд записать до входа в зону
- `postbuffer_seconds` — сколько секунд дописать после выхода
- `fps_fallback` — если runtime FPS еще не стабилизировался
- `codec` — FOURCC для writer
- `draw_recording_overlay` — рисовать ли поверх incident-кадров блок `ALERT RECORDING`

## Overlay

Теперь для tracked cats по умолчанию:

- большие bbox не рисуются;
- остается крестик цели;
- остается подпись `Cat N`;
- остается текст `zone: ...`

Это настраивается так:

```yaml
overlay:
  show_track_boxes: false
  show_track_crosshair: true
```

Если нужно вернуть коробки:

```yaml
overlay:
  show_track_boxes: true
```

Во время активной incident recording поверх кадра может рисоваться блок:

- `ALERT RECORDING`
- `now: ...`
- `zone: ...`
- `cats: ...`
- `started: ...`
- `elapsed: ...`

Он попадает и в live-window, и в incident-video.

## Полезные Логи

Основные runtime-сообщения:

- `track_id=... entered floor zone ...`
- `track_id=... entered surface zone ...`
- `track_id=... entered restricted zone ...`
- `track_id=... returned to floor zone ...`
- `surface_alert_emitted track_id=... zone=...`
- `surface_alert_audio_played track_id=... zone=... backend=...`
- `surface_alert_audio_failed track_id=... zone=... reason=...`
- `surface_alert_suppressed_by_cooldown track_id=... zone=...`
- `surface_alert_continuous_started track_id=... zone=...`
- `surface_alert_continuous_stopped`
- `alert_recording_started path=... zones=... tracks=...`
- `alert_recording_finished video=... duration=... zones=... tracks=...`

Типовые причины `surface_alert_audio_failed`:

- `sound_file_missing`
- `backend_unavailable`
- `playback_command_failed`

## Как Проверить, Что Все Работает

1. Убедись, что `scene_zones.enabled: true`.
2. Включи `scene_zones.draw_zones: true`.
3. Включи `scene_zones.draw_track_locations: true`.
4. Включи `surface_alert.enabled: true`.
5. Убедись, что на экране у кошки виден крестик.
6. Проверь, что alert не срабатывает, пока крестик еще вне `restricted`.
7. Когда крестик входит в `restricted`:
   - появляется `surface_alert_emitted ...`;
   - появляется `surface_alert_audio_played ...` или `surface_alert_audio_failed ...`;
   - при `continuous_while_in_zone: true` появляется `surface_alert_continuous_started ...`;
   - создается incident-video в `alert_video`.
8. После выхода всех кошек из alert-зон:
   - continuous alarm останавливается;
   - после `postbuffer_seconds` закрывается incident;
   - в папке остаются `.mp4` и `.json`.

## Тесты

Все тесты:

```bash
PYTHONPATH=. .venv/bin/pytest
```

Полезные выборки:

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_surface_monitor.py
PYTHONPATH=. .venv/bin/pytest tests/test_alert_recorder.py
PYTHONPATH=. .venv/bin/pytest tests/test_overlay.py
PYTHONPATH=. .venv/bin/pytest tests/test_zones.py
```

Что покрыто тестами:

- bbox edge inside zone but crosshair outside zone -> alert не срабатывает;
- crosshair inside restricted zone -> alert срабатывает;
- overlay boxes disabled -> bbox boxes не рисуются;
- incident recording стартует на первом alert;
- incident recording пишет кадры, пока alert активен;
- incident recording останавливается после конца alert + postbuffer;
- prebuffer попадает в видео;
- output dir создается автоматически;
- имя файла содержит timestamp + zone + track ids;
- multi-cat incident использует один session, пока хотя бы один кот остается в alert zone.


## Выбор Источника Камеры

Теперь проект умеет работать в двух режимах:

### 1. Встроенная webcam
По умолчанию используется `configs/default.yaml`:

```bash
.venv/bin/python -m app.main --config configs/default.yaml
```

### 2. ESP32 по Wi-Fi
Для ESP32 по умолчанию используется более лёгкая модель `yolo26n.pt`, чтобы обработка была живее на маленьком кадре `160x120`.

Для ESP32 snapshot-камеры добавлен отдельный конфиг:

```bash
.venv/bin/python -m app.main --config configs/esp32_wifi.yaml
```

В этом режиме приложение читает не RTSP и не MJPEG stream, а регулярно забирает JPEG с:

```text
http://192.168.8.140/snapshot.jpg
```

Под капотом для такого режима добавлен новый `source_type: http_snapshot`.

Можно запускать и через CLI без отдельного конфига:

```bash
.venv/bin/python -m app.main   --config configs/default.yaml   --source http_snapshot   --input http://192.168.8.140/snapshot.jpg   --snapshot-timeout 2.0
```

## Что Улучшено Для Стабильности

Сделаны практические улучшения:

- добавлен отдельный snapshot-source для ESP32 вместо попытки открыть HTML или JPEG через `cv2.VideoCapture`;
- добавлен cache-bust query, чтобы не залипать на старом кадре из кеша;
- добавлен таймаут HTTP snapshot-запроса;
- улучшен reconnect для Wi-Fi источника при пропаже кадра;
- исправлен путь incident recording в `configs/default.yaml` на текущий проект `TrackingCatWIFI`.


## Быстрый Выбор Камеры

Чтобы не помнить конфиги, добавлен быстрый launcher:

```bash
./run_camera.sh webcam
./run_camera.sh esp32
```

Он просто запускает нужный конфиг:
- `webcam` -> `configs/default.yaml`
- `esp32` -> `configs/esp32_wifi.yaml`

Можно пробрасывать дополнительные аргументы дальше в приложение:

```bash
./run_camera.sh esp32 --show-window true
./run_camera.sh webcam --camera-index 1
```


## Отдельные Зоны Для ESP32

ESP32 теперь использует свой собственный конфиг `configs/esp32_wifi.yaml` с отдельной секцией `scene_zones`.

Это значит:
- зоны основной камеры остаются в `configs/default.yaml`;
- зоны ESP32 живут отдельно и не собьют основную камеру;
- если запускать zone editor для ESP32 через `--config configs/esp32_wifi.yaml`, сохраняться будут только ESP32-зоны.

Для ESP32 overlay тоже упрощён:
- убран почти весь текст;
- оставлен крестик;
- оставлен счётчик кошек;
- подписи зон скрыты.


## Анти-ложные Срабатывания Для ESP32

Для ESP32 добавлена более жёсткая фильтрация детектора:
- `confidence_threshold: 0.5`
- `max_frame_area_ratio: 0.45`

Это помогает отсеивать ложные `cat`-детекции, когда объект занимает почти весь кадр, как в случае с лицом крупным планом на `160x120`.


## CPU И GPU Запуск

Теперь launcher умеет выбирать не только камеру, но и устройство inference.

### Быстрый запуск

```bash
./run_camera.sh webcam cpu
./run_camera.sh webcam gpu
./run_camera.sh esp32 cpu
./run_camera.sh esp32 gpu
```

Где:
- `cpu` -> запускает с `--device cpu`
- `gpu` -> запускает с `--device cuda:0`

### Прямой запуск без launcher

Основная камера на CPU:

```bash
.venv/bin/python -m app.main --config configs/default.yaml --device cpu
```

Основная камера на GPU:

```bash
.venv/bin/python -m app.main --config configs/default.yaml --device cuda:0
```

ESP32 на CPU:

```bash
.venv/bin/python -m app.main --config configs/esp32_wifi.yaml --device cpu
```

ESP32 на GPU:

```bash
.venv/bin/python -m app.main --config configs/esp32_wifi.yaml --device cuda:0
```

Важно: GPU-режим реально заработает только если `torch.cuda.is_available()` возвращает `True`.
Если CUDA недоступна, проект сам откатится на CPU и напишет предупреждение в лог.


## CPU-оптимизация

Для уменьшения лага и более живого трекинга были ужаты CPU-настройки.

### Основная камера
- `camera_width/camera_height`: `800x450`
- `imgsz`: `512`
- чаще запуск YOLO во время трекинга
- более агрессивное обновление треков, чтобы крестик меньше отставал
- overlay очищен: оставлены крестик, FPS и счётчик кошек

### ESP32
- `imgsz`: `320`
- YOLO проверяет кадр чаще, чтобы меньше было запаздывания
- сглаживание сделано более отзывчивым
- overlay очищен до минимума

Если захочешь ещё быстрее, следующий шаг это переход основной камеры на `yolo26n.pt` или ещё ниже по `imgsz`.


## Текущий Launcher

После чистки CUDA/NVIDIA пакетов launcher снова упрощён и запускает проект только в CPU-режиме:

```bash
./run_camera.sh webcam
./run_camera.sh esp32
```

При необходимости дополнительные аргументы можно пробросить дальше:

```bash
./run_camera.sh webcam --show-window true
./run_camera.sh esp32 --zone-editor true
```
GPU/ROCm запуск сейчас не используется.


## CPU-only окружение

Из `.venv` удалены лишние CUDA/NVIDIA пакеты, которые только занимали место.
Теперь установлен CPU-only `torch`:
- `torch==2.11.0+cpu`
- `torchvision==0.26.0+cpu`


## Multi-cat tweak for webcam

Для основной камеры профиль дополнительно смещён в сторону лучшего захвата нескольких кошек:
- `imgsz` возвращён на `640`
- снижен `confidence_threshold`
- YOLO снова проверяет каждый кадр во время трекинга
- подтверждение нового трека ускорено
- окна reacquire/hold/lost расширены
- loosened IoU/area/distance gates для маленьких и частично перекрытых кошек
- `smoothing_alpha` увеличен, чтобы крестик быстрее догонял живой объект

Это сделано специально для сцены, где одна кошка ближе, а вторая меньше/дальше и может теряться.


## Режим безопасного наведения

Для сценария будущего наведения по цели логика ужесточена:
- рабочий крестик и отображаемая цель показываются только по свежему YOLO-подтверждению текущего кадра;
- tracker-only хвост больше не может уехать в сторону и остаться рабочей целью;
- если свежего детекта нет, цель лучше кратко пропадёт, чем сместится с кошки.

Это специально сделано в пользу точности наведения, а не непрерывной красивой визуализации.


## Логи runtime

Основной runtime log теперь всегда пишется сюда:

```text
/home/kelbus/PycharmProjects/TrackingCatWIFI/log.txt
```

При каждом новом запуске файл очищается и начинается заново.


## Professional fail-safe aiming mode

Для основного профиля наведения включён более строгий принцип, как это обычно делают в safety-critical computer vision workflow:

- OpenCV frame-tracker не используется как самостоятельный источник истины для рабочей цели;
- рабочая цель берётся только из свежих YOLO detections текущего кадра;
- track IDs и ассоциация между кадрами сохраняются, но без tracker-only дрейфа;
- если свежего подтверждения нет, цель лучше пропадает, чем уезжает с кошки.

Именно это поведение безопаснее для будущего наведения по цели.


## Production profiles

Проект теперь разделён на два разных профиля для основной камеры.

### 1. `webcam_aiming_safe.yaml`
Это основной production-профиль для будущего наведения.

Принцип:
- только свежие YOLO detections считаются рабочей целью;
- OpenCV frame-tracker не используется как источник истины;
- лучше кратко потерять цель, чем сместить её с кошки.

Запуск:

```bash
./run_camera.sh webcam-safe
```

### 2. `webcam_visual_tracking.yaml`
Это отдельный визуальный режим, если нужен более плавный display/tracking-preview.

Принцип:
- может использовать tracker-only bridge между detections;
- визуально плавнее;
- но не предназначен как fail-safe режим для наведения.

Запуск:

```bash
./run_camera.sh webcam-visual
```

### 3. `esp32_wifi.yaml`
Отдельный профиль ESP32.

Запуск:

```bash
./run_camera.sh esp32
```

### Рекомендация
Для реального production use и будущего наведения использовать именно:

```bash
./run_camera.sh webcam-safe
```


## Browser Camera (phone / tablet / laptop)

Теперь проект умеет принимать кадры прямо из браузера другого устройства в локальной сети.

Сценарий:

1. На iMac запускается ingest-страница:

```bash
cd ~/PycharmProjects/TrackingCatWIFI
./run_camera.sh browser
```

2. С любого устройства в той же сети открыть:

```
http://192.168.8.176:8020/
```

3. Нажать **Start camera** и разрешить доступ к камере.

После этого `TrackingCatWIFI` начнет использовать кадры из браузера как live-source.

Что сейчас реализовано:

- встроенный HTTP ingest сервер внутри `TrackingCatWIFI`;
- web-страница захвата камеры;
- отправка JPEG кадров на сервер;
- отдельный конфиг `configs/browser_camera.yaml`;
- отдельные зоны для browser camera, независимые от webcam и ESP32.

Полезные URL:

- ingest page: `http://192.168.8.176:8020/`
- health: `http://192.168.8.176:8020/health`
- latest uploaded frame: `http://192.168.8.176:8020/snapshot.jpg`

Важно:

- это минимальный LAN MVP без настоящего WebRTC стрима;
- сервер получает последовательные JPEG кадры;
- если Safari/Chrome на телефоне откажется давать камеру на обычном `http`, следующий шаг, поднять локальный `https` для той же страницы.

Редактор зон для browser camera:

```bash
cd ~/PycharmProjects/TrackingCatWIFI
.venv/bin/python -m app.main --config configs/browser_camera.yaml --zone-editor true --device cpu
```


## iPhone via IP Camera Lite

Рабочий локальный источник с iPhone через MJPEG:

- URL: `http://192.168.8.181:8081/video`
- config: `configs/iphone_ipcamera.yaml`

Запуск:

```bash
cd ~/PycharmProjects/TrackingCatWIFI
./run_camera.sh iphone
```

Редактор зон для iPhone-камеры:

```bash
cd ~/PycharmProjects/TrackingCatWIFI
.venv/bin/python -m app.main --config configs/iphone_ipcamera.yaml --zone-editor true --device cpu
```

Важно:
- iPhone и iMac должны быть в одной Wi-Fi сети.
- В IP Camera Lite должен быть включен live server mode.
- Если IP телефона поменяется, обнови `source.stream_url` в `configs/iphone_ipcamera.yaml`.
