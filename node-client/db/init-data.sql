INSERT INTO users (
    first_name,
    last_name,
    username,
    password
)
VALUES
    ('Lior', 'Ohana', 'LiorOhana94','$2b$10$k7dsUc2aLcKjkS0CTCuuqOvYjDnO4jZNQWWtKSn37USBCGyvLHHlS'),
    ('Lior', 'Eliav', 'LiorEliav', '$2b$10$k7dsUc2aLcKjkS0CTCuuqOvYjDnO4jZNQWWtKSn37USBCGyvLHHlS'),
    ('Amir', 'Kirsh', 'AmirTheKing', '$2b$10$k7dsUc2aLcKjkS0CTCuuqOvYjDnO4jZNQWWtKSn37USBCGyvLHHlS');

INSERT INTO patients (
    first_name,
    last_name,
    age,
    phone_number
)
VALUES
    ('Shalom', 'Aleichem', 52, '0548732111'),
    ('Haim', 'Cohen', 68, '0528899456'),
    ('Gila', 'Gamliel', 90, '0501111112');

INSERT INTO diagnosis (name)
VALUES ("Unknown"), ("Positive"), ("Negative");

