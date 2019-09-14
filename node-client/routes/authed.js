const express = require('express');
const router = express.Router();
const uuid = require('uuid/v4');
const fetch = require('node-fetch');
const fs = require('fs');
const path = require('path');
const multer = require('multer');
const diagnosisHandler = require('../activations-downloader');

const upload = multer({
    dest: "./temp/"
});


var db;
(async function () {
    db = await require('../db/db');
})();

function validateUser(req, res, next) {
    if (req.user) {
        next();
    } else {
        res.redirect('/login');
    }
}


router.use(validateUser);

router.get('/send', function (req, res, next) {
    res.render('send');
})


router.get('/', function (req, res, next) {
    res.render('index', {
        user: {
            firstname: req.user.first_name,
            lastname: req.user.last_name
        }
    });
});

router.get('/aboutPage', function (req, res, next) {
    res.render('about', {
        user: {
            firstname: req.user.first_name,
            lastname: req.user.last_name
        }
    });
});

router.get('/uploadPage', function (req, res, next) {
    res.render('upload', {
        user: {
            firstname: req.user.first_name,
            lastname: req.user.last_name
        }
    });
});


router.post('/doesIdExist', async function (req, res, next) {
    console.log(req.body);
    const [results, tableDef] = await db.execute(`Select * from patients where patient_id="${req.body.id}";`);
    if (results.length === 0) {
        res.status(200).send(false);
    } else {
        res.status(200).send(true);
    }
});

router.post('/getScans', async function (req, res, next) {
    const [results, tableDef] = await db.execute(`Select first_name as firstName, last_name as lastName, scans.id as scanId, date, s_diagnosis.name as 'systemDiagnosis', f_diagnosis.name as 'finalDiagnosis'
from patients 
inner join scans on patients.id = scans.patient_id
left join diagnosis s_diagnosis on s_diagnosis.id = scans.system_diagnosis_id
left join diagnosis f_diagnosis on f_diagnosis.id = scans.final_diagnosis_id
where patients.patient_id = ${req.body.patientId};`);

    if (results.length === 0) {
        res.status(200).send([]);
    } else {
        results.forEach(r => r.date = formatDate(r.date));
        res.status(200).send(results);
    }
});

router.post('/addPatient', async function (req, res, next) {
    console.log(req.body);
    const [results, tableDef] = await db.execute(`INSERT INTO patients (patient_id, first_name, last_name, age, phone_number) VALUES (${req.body.id}, "${req.body.firstName}", "${req.body.lastName}", "${req.body.age}", "${req.body.phone}");`);
    res.status(200).send(true);
});

router.post('/updateFinalDiagnosis', async function (req, res, next) {
    console.log(req.body);
    let value = req.body.finalDiagnosis ? 1 : 2;
    const [results, tableDef] = await db.execute(`UPDATE scans SET final_diagnosis_id=${value} WHERE id=${req.body.scanID};`);
    res.status(200).send(true);
});


router.get('/searchPage', function (req, res, next) {
    res.render('search', {
        user: {
            firstname: req.user.first_name,
            lastname: req.user.last_name
        }
    });
});


router.get('/reviewDiagnosis/:scanId', async function (req, res, next) {

    let [results, tableDef] = await db.execute(`select scans.id as id, system_diagnosis_id, patients.patient_id as patientId, first_name as patientFirstname, last_name as patientLastname, map_guid, date from scans left join patients on patients.id = scans.patient_id where scans.id = ${req.params.scanId};`);

    if (results.length === 0) {
        res.status(405);
        return;
    }

    let resultIndex = (function (sdi) {
        if (!sdi) {
            return null
        } else if (sdi === 1) {
            return 'pneumonia';
        } else {
            return 'healty';
        }
    })(results[0].system_diagnosis_id);

    let resultText = (function (sdi) {
        if (!sdi) {
            return 'none'
        } else if (sdi === 1) {
            return 'pneumonia';
        } else {
            return 'healty';
        }
    })(results[0].system_diagnosis_id);

    let locals = {
        data: {
            "heatmap_guid": results[0].map_guid,
            "result_index": resultIndex,
            "result_text": resultText,
            "scan_id": results[0].id,
            "patient_id": results[0].patientId,
            "patient_firstname": results[0].patientFirstname,
            "patient_lastname": results[0].patientLastname,
            "date": formatDate(results[0].date)
        }
    };
    res.render('diagnosisReview', locals);
});


router.post("/diagnoseScan",
    function (req, res, next) {
        next();
    },
    upload.single("scan"),
    async function (req, res, next) {
        try {

            if (!req.file) {
                res.status(405);
            }

            const tempPath = req.file.path;
            const ext = path.extname(req.file.originalname).toLowerCase();
            const filename = uuid();
            const targetPath = path.join(__dirname, `../public/scans/${filename}${ext}`);
            if (ext === ".png" || ext === '.jpeg' || ext === '.jpg') {
                fs.rename(tempPath, targetPath, async err => {
                    if (err) {
                        console.log(err);
                        return;
                    }
                    let [results, tableDef] = await db.execute(`SELECT id, first_name, last_name FROM patients WHERE patient_id=${req.body.patientId}`);

                    if (results.length === 0) {
                        console.log("could not find patient");
                        res.status(405);
                        return;
                    }

                    req.patientFirstname = results[0]['first_name'];
                    req.patientLastname = results[0]['last_name'];

                    results = await db.execute(`INSERT INTO scans (patient_id, file_name) VALUES(${results[0].id}, '${filename + ext}')`);

                    req.filename = filename + ext;
                    req.scanId = results[0].insertId;
                    req.scanDate = new Date();
                    req.scanGuid = filename;

                    next();
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
                        .end("Only .png / .jpeg / .jpg files are allowed!");
                });

            }
        } catch (e) {
            console.log(e);
        }
    },
    async function (req, res, next) {
        await fetch(global.nnEndpoint + '/predict', {
            method: 'post',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                scan: `${global.config.endpoint}/scans/${req.filename}`,
                scanGuid: req.scanGuid
            })
        });

        fetch(global.nnEndpoint + '/predict', {
            method: 'post',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                scan: `${global.config.endpoint}/scans/${req.filename}`,
                scanGuid: req.scanGuid
            })
        }).then(nnRes => {
            let json = nnRes.json();
            json.then((data) => {
                req.nnResponse = data;
                console.log("Response from NN: ", data);
                next();
                if (data.heatmap_guid) {
                    diagnosisHandler.emit('mapGenerated', data.heatmap_guid);
                }
            })
        })
    },

    async function (req, res, next) {

        await db.execute(`UPDATE scans set system_diagnosis_id=${req.nnResponse['result_text'] === 'pneumonia' ? 1 : 2} where id=${req.scanId}`);

        let locals = {
            data: {
                "heatmap_guid": req.nnResponse['heatmap_guid'],
                "result_index": req.nnResponse['result_index'],
                "result_prob": req.nnResponse['result_prob'],
                "result_text": req.nnResponse['result_text'],
                "scan_id": req.scanId,
                "patient_id": req.body.patientId,
                "patient_firstname": req.patientFirstname,
                "patient_lastname": req.patientLastname,
                "date": formatDate(req.scanDate)
            }
        };
        console.log(locals);
        res.render('diagnosisReview', locals);
    }
);

router.get('/updateEndpoint', function (req, res, next) {
    if (req.user.id > 2) {
        res.send(401);
    } else {
        next();
    }
}, function (req, res, next) {
    res.render('updateEndpoint', {success: false});
});

router.post('/updateEndpoint', function (req, res, next) {
    if (req.user.id > 2) {
        res.send(401);
    } else {
        next();
    }
}, async function (req, res, next) {
    let nnEndpoint = 'http://' + req.body.endpoint + ':8080';

    await db.execute(`UPDATE endpoint SET domain_string="${nnEndpoint}" where id=1;`);
    global.nnEndpoint = nnEndpoint;

    res.render('updateEndpoint', {success: true});
});


function formatDate(date) {
    return `${date.getDate()}/${date.getMonth() + 1}/${date.getFullYear()}`
}


module.exports = router;
