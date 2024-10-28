CREATE TABLE [Guest] (
  [guest_id] int PRIMARY KEY IDENTITY(1, 1),
  [name] varchar(100),
  [email] varchar(100),
  [phone] varchar(15)
)
GO

CREATE TABLE [RoomType] (
  [room_type_id] int PRIMARY KEY IDENTITY(1, 1),
  [name] varchar(50),
  [bed_count] int,
  [price_per_night] decimal(10,2)
)
GO

CREATE TABLE [Room] (
  [room_id] int PRIMARY KEY IDENTITY(1, 1),
  [room_type_id] int,
  [room_number] varchar(10),
  [is_occupied] boolean
)
GO

CREATE TABLE [Reservation] (
  [reservation_id] int PRIMARY KEY IDENTITY(1, 1),
  [guest_id] int,
  [room_id] int,
  [employee_id] int,
  [creation_date] date,
  [check_in_date] date,
  [check_out_date] date,
  [status] varchar(20)
)
GO

CREATE TABLE [Employee] (
  [employee_id] int PRIMARY KEY IDENTITY(1, 1),
  [name] varchar(100),
  [position] varchar(50)
)
GO

CREATE TABLE [Expense] (
  [expense_id] int PRIMARY KEY IDENTITY(1, 1),
  [reservation_id] int,
  [description] varchar(255),
  [amount] decimal(10,2),
  [days] int
)
GO

CREATE TABLE [Payment] (
  [payment_id] int PRIMARY KEY IDENTITY(1, 1),
  [reservation_id] int,
  [total_accommodation] decimal(10,2),
  [total_expenses] decimal(10,2),
  [payment_date] date,
  [is_paid] boolean
)
GO

ALTER TABLE [Room] ADD FOREIGN KEY ([room_type_id]) REFERENCES [RoomType] ([room_type_id])
GO

ALTER TABLE [Reservation] ADD FOREIGN KEY ([guest_id]) REFERENCES [Guest] ([guest_id])
GO

ALTER TABLE [Reservation] ADD FOREIGN KEY ([room_id]) REFERENCES [Room] ([room_id])
GO

ALTER TABLE [Reservation] ADD FOREIGN KEY ([employee_id]) REFERENCES [Employee] ([employee_id])
GO

ALTER TABLE [Expense] ADD FOREIGN KEY ([reservation_id]) REFERENCES [Reservation] ([reservation_id])
GO

ALTER TABLE [Payment] ADD FOREIGN KEY ([reservation_id]) REFERENCES [Reservation] ([reservation_id])
GO
