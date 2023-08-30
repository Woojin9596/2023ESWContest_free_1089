import serial
import pynmea2
import folium

# 시리얼 포트와 속도 설정
ser = serial.Serial('/dev/ttyAMA0', 9600, timeout=1)

# 초기화한 folium 지도 객체 생성
m = folium.Map(location=[37.2974, 126.837], zoom_start=12)

def read_gps_data():
    try:
        line = ser.readline().decode('utf-8')
        if line.startswith('$GPGGA'):
            msg = pynmea2.parse(line)
            latitude = msg.latitude
            longitude = msg.longitude
            return latitude, longitude
    except Exception as e:
        print("Error:", e)
        return None, None

def plot_gps_on_map(latitude, longitude):
    if latitude is not None and longitude is not None:
        # 지도에 점 찍기
        folium.Marker([latitude, longitude], popup='GPS Point').add_to(m)
        # 지도를 HTML 파일로 저장
        m.save('map_with_gps.html')

# 메인 루프
def main():
    while True:
        latitude, longitude = read_gps_data()
        plot_gps_on_map(latitude, longitude)

if __name__ == "__main__":
    main()