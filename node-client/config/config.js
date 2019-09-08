module.exports = {
  development: {
    host: 'localhost',
    user: 'root',
    password: 'Lior265840',
    database: 'medikals',
    secret: 'grumpy_pink_cat',
    endpoint: 'http://192.168.43.182:3000',
    nnEndpoint: 'http://192.168.43.82:8080'
  },
  lilidev: {
    host: 'localhost',
    user: 'root',
    password: '1234',
    database: 'medikals',
    secret: 'grumpy_yello_cat',
    endpoint: 'http://192.168.43.182:3000',
    nnEndpoint: 'http://192.168.43.82:8080'
  },
  production: {
    host: 'localhost',
    user: 'root',
    password: 'eliohana',
    database: 'medikals',
    endpoint: 'http://ec2-52-201-234-115.compute-1.amazonaws.com',
    secret: 'grumpy_blue_cat'
  }
}