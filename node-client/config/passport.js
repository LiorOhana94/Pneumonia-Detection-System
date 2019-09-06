var db ;
(async function(){
    db = await require('../db/db');
})();

const LocalStrategy = require('passport-local').Strategy;
const bcrypt = require('bcrypt');

module.exports = function (passport) {
    passport.serializeUser((user, done) => {
        done(null, user.id);
    });

    passport.deserializeUser(async (id, done) => {
        try {
            const [results, tableDef] = await db.execute(`Select * from users where id=${id};`);
            done(null, results[0]);
        }
        catch (err) {
            done(err);
        }
    });

    passport.use('local', new LocalStrategy(
        async (username, password, done) => {
            try {
                const [results, tableDef] = await db.execute(`Select * from users where username="${username}";`);
                if (results.length === 0) {
                    return done(null, false);
                }

                bcrypt.compare(password, results[0].password, (err, res) => {
                    if (err) {
                        return done(err);
                    }
                    if (!res) {
                        return done(null, false);
                    }

                    return done(null, results[0]);
                });
            } catch (err) {
                done(err);
            }
        }));
};