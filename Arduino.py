// Define pump control pins
const int PUMP1_PIN = 2;
const int PUMP2_PIN = 3;
const int PUMP3_PIN = 4;
const int PUMP4_PIN = 5;

void setup() {
  Serial.begin(9600);
  
  // Set all pump pins as outputs
  pinMode(PUMP1_PIN, OUTPUT);
  pinMode(PUMP2_PIN, OUTPUT);
  pinMode(PUMP3_PIN, OUTPUT);
  pinMode(PUMP4_PIN, OUTPUT);
  
  // Start with all pumps OFF
  digitalWrite(PUMP1_PIN, LOW);
  digitalWrite(PUMP2_PIN, LOW);
  digitalWrite(PUMP3_PIN, LOW);
  digitalWrite(PUMP4_PIN, LOW);
  
  Serial.println("7-Minute Pump Cycle Starting!");
  Serial.println("Cycle: Pump1(1min) -> Wait(2min) -> Pump2(1min) -> Wait(2min) -> Pump3(1min)");
}

void loop() {
  // Minute 1: Pump 1 ON for 1 minute
  Serial.println("Minute 1: Pump 1 ON");
  digitalWrite(PUMP1_PIN, HIGH);
  delay(60000); // 60 seconds = 1 minute
  digitalWrite(PUMP1_PIN, LOW);
  Serial.println("Pump 1 OFF");
  
  // Minutes 2-3: Wait 2 minutes (nothing on)
  Serial.println("Minutes 2-3: Waiting...");
  delay(120000); // 120 seconds = 2 minutes
  
  // Minute 4: Pump 2 ON for 1 minute
  Serial.println("Minute 4: Pump 2 ON");
  digitalWrite(PUMP2_PIN, HIGH);
  delay(60000); // 1 minute
  digitalWrite(PUMP2_PIN, LOW);
  Serial.println("Pump 2 OFF");
  
  // Minutes 5-6: Wait 2 minutes
  Serial.println("Minutes 5-6: Waiting...");
  delay(120000); // 2 minutes
  
  // Minute 7: Pump 3 ON for 1 minute
  Serial.println("Minute 7: Pump 3 ON");
  digitalWrite(PUMP3_PIN, HIGH);
  delay(60000); // 1 minute
  digitalWrite(PUMP3_PIN, LOW);
  Serial.println("Pump 3 OFF");
  
  // start the thing again
  Serial.println("Cycle complete! Restarting in 3 seconds...");
  delay(3000);
  Serial.println("-----------------------------------");
}

