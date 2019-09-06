var express = require('express');
var router = express.Router();
var db;
(async function () {
  db = await require('../db/db');
})();

function validateUser(req, res, next) {
  if (req.user) {
    next();
  }
  else {
    res.redirect('/login');
  }
}

router.use(validateUser);

router.get('/', function (req, res, next) {
  res.render('index');
});

router.get('/aboutPage', function (req, res, next) {
  res.render('about');
});

router.get('/uploadPage', function (req, res, next) {
  res.render('upload');
});

router.post('/doesIdExist', async function (req, res, next) {
  console.log(req.body);
  const [results, tableDef] = await db.execute(`Select * from patients where patient_id="${req.body.id}";`);
  if (results.length === 0) {
    res.status(200).send(false);
  }
  else {
    res.status(200).send(true);
  }
});

router.post('/getScan', async function (req, res, next) {
  console.log(req.body);
  const [results, tableDef] = await db.execute(`Select * from scans where id="${req.body.scanId}";`);
  if (results.length === 0) {
    res.status(200).send({ found: false });
  }
  else {
    res.status(200).send({
      found: true,
      scanData: results[0]
    });
  }
});

router.post('/addPatient', async function (req, res, next) {
  console.log(req.body);
  const [results, tableDef] = await db.execute(`INSERT INTO patients (patient_id, first_name, last_name, age, phone_number) VALUES (${req.body.id}, "${req.body.firstName}", "${req.body.lastName}", "${req.body.age}", "${req.body.phone}");`);
  res.status(200).send(true);
});

router.post('/dignoseImage', function (req, res, next) {
  console.log(req.body);
});

router.get('/searchPage', function (req, res, next) {
  res.render('search');
});

module.exports = router;


const fs = require('fs');
const path = require('path');
var multer  = require('multer');

const upload = multer({
  dest: "./temp/"
  // you might also want to set some limits: https://github.com/expressjs/multer#limits
});

router.post(
  "/upload-image",
  upload.single("scanToDiagnose"),
  (req, res) => {
    try {
      const tempPath = req.file.path;
      const targetPath = path.join(__dirname, "../upload/image.png");

      if (path.extname(req.file.originalname).toLowerCase() === ".png") {
        fs.rename(tempPath, targetPath, err => {
          if (err) {
            console.log(err);
            return;
          }

          res
            .status(200)
            .contentType("text/plain")
            .end("File uploaded!");
        });
      } else {
        fs.unlink(tempPath, err => {
          if (err) {
            console.log(err);
            return;
          }

          res
            .status(403)
            .contentType("text/plain")
            .end("Only .png files are allowed!");
        });

      }
    } catch (e) {
      console.log(e);
    }
  }
);