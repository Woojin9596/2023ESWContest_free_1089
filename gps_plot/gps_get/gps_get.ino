#include <SoftwareSerial.h>

SoftwareSerial gpsSerial(11, 12);

void setup(){
  Serial.begin(9600);
  Serial.println("Start GPS... ");
  gpsSerial.begin(9600);
}

void loop(){
  if(gpsSerial.available()){
    Serial.write(gpsSerial.read());
  }
}