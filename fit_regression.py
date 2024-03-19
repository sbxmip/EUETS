import os
import swat
import getpass

password = getpass.getpass ('Enter your password : ')
hostname = "https://viya4-s2.zeus.sashq-d.openstack.sas.com/"
os.environ["CAS_CLIENT_SSL_CA_LIST"] = r"C:\Users\sbxmip\.vscode\trustedcerts.pem"
conn = swat.CAS('https://viya4-s2.zeus.sashq-d.openstack.sas.com/cas-shared-default-http', username='sbxmip',password=password)
EU_ETS_tbl = conn.CASTable('EU_ETS_CURATED', caslib='EBA')

conn.loadActionSet('regression')
conn.regression.glm( 
    table = dict(name = EU_ETS_tbl, where = 'year_comp = 2021'), 
    classVars=['Country_id','nace_desc','activity_desc'],
    model={'depVar':'verified',
                   'effects':['Country_id', 'activity_desc', 'nace_desc']
                  },
    store = dict(name='reg_model', replace=True),
    selection = {"method":"FORWARD"}
)

