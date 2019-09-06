const mysql = require('mysql2/promise');
 
// create the connection to database
const connection = mysql.createConnection({
  host: global.config.host,
  user: global.config.user,
  password: global.config.password,
  database: global.config.database
});

module.exports = connection;