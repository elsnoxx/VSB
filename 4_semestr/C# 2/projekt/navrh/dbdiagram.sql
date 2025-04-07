// Table for parking lot
Table ParkingLot {
  parking_lot_id INT [pk]
  name VARCHAR(255)
  gps_coordinates POINT
}

// Table for parking space
Table ParkingSpace {
  parking_space_id INT [pk]
  parking_lot_id INT
  space_number INT
  status ENUM('available', 'occupied', 'under_maintenance')
}

// Table for occupancy
Table Occupancy {
  occupancy_id INT [pk]
  parking_space_id INT
  license_plate VARCHAR(20)
  start_time DATETIME
  end_time DATETIME
  duration INT
  price DECIMAL(10, 2)
}

// Table for parking space status history
Table StatusHistory {
  history_id INT [pk]
  parking_space_id INT
  status ENUM('available', 'occupied', 'under_maintenance')
  change_time DATETIME
}

// Table for statistics
Table Statistics {
  parking_lot_id INT
  month DATE
  completed_parkings INT
}

// Relationships
Ref: ParkingSpace.parking_lot_id > ParkingLot.parking_lot_id
Ref: Occupancy.parking_space_id > ParkingSpace.parking_space_id
Ref: StatusHistory.parking_space_id > ParkingSpace.parking_space_id
Ref: Statistics.parking_lot_id > ParkingLot.parking_lot_id
