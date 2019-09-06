const eventEmitter = require('events');
const uuid = require('uuid/v4');
const fetch = require('node-fetch');

var db;
(async function () {
    db = await require('./db/db');
})();

class DiagnosisHandler extends eventEmitter {}

const diagnosisHandler = new DiagnosisHandler();


diagnosisHandler.on('scanComplete', async function (activationMapGuid) {
    const guid = uuid();

    await db.execute(`INSERT INTO upload_hashes (hash) values("${guid}")`);

    fetch(`${global.config.nnEndpoint}/send-activation-map`, {
        method: 'post',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            scanGuid: activationMapGuid,
            destination: `${global.config.endpoint}/upload-map/${guid}`
        })
    }).then(function (req) {
        console.log(req);
    })

});

module.exports = diagnosisHandler;