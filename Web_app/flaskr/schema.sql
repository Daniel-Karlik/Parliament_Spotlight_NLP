DROP TABLE IF EXISTS user;
DROP TABLE IF EXISTS post;

CREATE TABLE user (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE NOT NULL,
  password TEXT NOT NULL
);

CREATE TABLE post (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  author_id INTEGER NOT NULL,
  created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  title TEXT NOT NULL,
  body TEXT NOT NULL,
  FOREIGN KEY (author_id) REFERENCES user (id)
);

CREATE TABLE steno_zaznamy (
  id TEXT NOT NULL,
  obdobi INTEGER,
  datum TEXT,
  schuze INTEGER,
  celeJmeno TEXT,
  OsobaId TEXT,
  funkce TEXT,
  tema TEXT,
  prsolovText TEXT,
  pocetSlov INTEGER,
  politiciZminky TEXT
);
