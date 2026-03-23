import cv2


def info_display(frame, time_ms):
    cv2.putText(frame, f"Time: {time_ms:.2f} ms", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    cv2.imshow('Analýza', frame)
    

def draw_eye(frame,x,y,ex,ey,ew,eh,state):
    cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 255), 2)
    cv2.putText(frame, f"{state}", (x+ex, y+ey-5),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)