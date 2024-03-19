EU_ETS_tbl = conn.CASTable('EU_ETS_CURATED', caslib='EBA')

conn.loadActionSet('regression')
conn.regression.glm( 
    table = dict(name = EU_ETS_tbl, where = 'year_comp = 2021'), 
    classVars=['Country_id','nace_desc','activity_desc'],
    model={'depVar':'verified',
                   'effects':['Country_id', 'activity_desc', 'nace_desc']
                  },
    store = dict(name='reg_model', replace=True),
    selection = {"method":"BACKWARD"}
)