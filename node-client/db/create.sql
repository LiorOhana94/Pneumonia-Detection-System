CREATE TABLE users (
    id INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    username VARCHAR(50) NOT NULL,
    profile_pic VARCHAR(100),
    password VARCHAR(255) NOT NULL 
);

CREATE TABLE patients (
    id INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    patient_id INT  UNIQUE,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    age INT NOT NULL,
    phone_number VARCHAR(12) NOT NULL 
);

CREATE TABLE diagnosis (
    id INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(50) NOT NULL
);

CREATE TABLE scans (
    id INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    patient_id INT NOT NULL,
    file_name VARCHAR(100) NOT NULL,
    date TIMESTAMP DEFAULT NOW() NOT NULL,
    system_diagnosis_id INT,
    final_diagnosis_id INT,

    FOREIGN KEY (patient_id) REFERENCES patients(id),
    FOREIGN KEY (system_diagnosis_id) REFERENCES diagnosis(id),
    FOREIGN KEY (final_diagnosis_id) REFERENCES diagnosis(id)
);

CREATE TABLE upload_hashes (
    id INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    hash VARCHAR(50) NOT NULL,
    used BIT(1) DEFAULT 0
);
