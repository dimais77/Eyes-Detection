import cv2

video = cv2.VideoCapture(0)
video.set(3, 600)
video.set(4, 600)
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

while True:
    success, frame = video.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=2, minSize=(10, 10))

    # Инициализируем переменные для хранения координат единой прямоугольной области
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = 0, 0

    # Находим минимальные и максимальные координаты для всех областей глаз
    for (x, y, w, h) in eyes:
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    # Добавляем смещение к координатам прямоугольной области
    offset = 30  # Выбераем значение для увеличени я области
    min_x -= offset
    min_y -= 0
    max_x += offset
    max_y += 0

    # Создаем прямоугольную область, охватывающую глаза
    eyes_roi = frame[min_y:max_y, min_x:max_x]

    # На случай, если "глаза потеряются" )))
    if not eyes_roi.size:
        continue

    # Размываем область глаз
    blurred_roi = cv2.GaussianBlur(eyes_roi, (45, 15), 20)

    # Заменяем прямоугольную область в кадре на размытую область
    frame[min_y:max_y, min_x:max_x] = blurred_roi

    # Отображаем кадр с размытой областью глаз
    cv2.imshow('Eyes Detection', frame)

    # Нажатие клавиши q для выхода из цикла
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
