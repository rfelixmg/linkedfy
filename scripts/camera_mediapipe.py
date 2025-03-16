import cv2
import pygame
import mediapipe as mp


def main():
    # Init camera and PyGame
    cap = cv2.VideoCapture(0)
    pygame.init()

    width, height = 640, 480
    screen = pygame.display.set_mode((width, height + 90))  # Space for buttons
    pygame.display.set_caption("Camera with Face Detection and Landmarks")

    font = pygame.font.SysFont(None, 30)

    face_button_rect = pygame.Rect(10, height + 10, 200, 30)
    landmark_button_rect = pygame.Rect(10, height + 50, 200, 30)

    detect_faces = False
    detect_landmarks = False

    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Mediapipe face mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face detection (rectangle)
        if detect_faces:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Mediapipe landmarks
        if detect_landmarks:
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for lm in face_landmarks.landmark:
                        x, y = int(lm.x * width), int(lm.y * height)
                        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        # Convert to RGB for PyGame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))

        # Display video
        screen.blit(frame_surface, (0, 0))

        # Draw buttons
        pygame.draw.rect(screen, (255, 0, 0), face_button_rect)
        face_text = "Face Detection: ON" if detect_faces else "Face Detection: OFF"
        screen.blit(font.render(face_text, True, (255, 255, 255)), (15, height + 15))

        pygame.draw.rect(screen, (0, 128, 0), landmark_button_rect)
        landmark_text = "Landmarks: ON" if detect_landmarks else "Landmarks: OFF"
        screen.blit(font.render(landmark_text, True, (255, 255, 255)), (15, height + 55))

        pygame.display.update()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if face_button_rect.collidepoint(event.pos):
                    detect_faces = not detect_faces
                if landmark_button_rect.collidepoint(event.pos):
                    detect_landmarks = not detect_landmarks

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()
