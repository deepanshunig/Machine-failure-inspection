-- Run manually in MySQL Workbench with an administrator account.
-- Replace the password placeholder locally. Do not save your real password in Git.
-- If the application account already exists, use its current password in .env.
CREATE DATABASE IF NOT EXISTS machine_failure_dashboard CHARACTER SET utf8mb4;
CREATE USER 'machine_failure_app'@'127.0.0.1' IDENTIFIED BY 'REPLACE_WITH_A_STRONG_PASSWORD';
GRANT ALL PRIVILEGES ON machine_failure_dashboard.* TO 'machine_failure_app'@'127.0.0.1';
-- Privileges are restricted to this project database. They permit schema migration.
