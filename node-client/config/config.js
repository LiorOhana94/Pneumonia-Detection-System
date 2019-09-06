module.exports = {
  development: {
    host: 'localhost',
    user: 'root',
    password: 'Lior265840',
    database: 'medikals',
    secret: 'grumpy_pink_cat',
    endpoint: 'http://10.0.0.4',
    nnEndpoint: 'http://localhost:8080'
  },
  lilidev: {
    host: 'localhost',
    user: 'root',
    password: '1234',
    database: 'medikals',
    secret: 'grumpy_yello_cat',
    endpoint: 'http://10.0.0.4:3000',
    nnEndpoint: 'http://10.0.0.5:8080'
  },
  production: {
    host: 'localhost',
    user: 'root',
    password: 'eliohana',
    database: 'medikals',
    endpoint: 'http://10.0.0.4',
    secret: 'grumpy_blue_cat'
  }
}