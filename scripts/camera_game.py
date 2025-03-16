import cv2
import pygame


def main():
    # Init camera and PyGame
    cap = cv2.VideoCapture(0)
    pygame.init()

    width, height = 1220, 880
    screen = pygame.display.set_mode((width, height + 50))  # Extra space for button
    pygame.display.set_caption("Camera with Face Detection Button")

    font = pygame.font.SysFont(None, 30)
    button_rect = pygame.Rect(10, height + 10, 200, 30)
    detect_faces = False  # Toggle flag

    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))

        # Face detection if toggled
        if detect_faces:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Convert frame to RGB for PyGame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))

        # Draw video
        screen.blit(frame_surface, (0, 0))

        # Draw button
        pygame.draw.rect(screen, (255, 0, 0), button_rect)
        button_text = "Face Detection: ON" if detect_faces else "Face Detection: OFF"
        text_render = font.render(button_text, True, (255, 255, 255))
        screen.blit(text_render, (15, height + 15))

        pygame.display.update()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    detect_faces = not detect_faces  # Toggle face detection

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()
