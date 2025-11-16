from flask import Flask,render_template,request,url_for,redirect

app=Flask(__name__)

# jijnja 2 template to read the values from backend
'''
1.{{}} expression print output
{%..}conditional statement
{#..#} sigle line comment
'''

@app.route("/")
def welcome():
    return "Welcome to this flask course,This is amazing course"

@app.route("/index")
def index():
    return render_template('index.html')
@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/form",methods=['GET','POST'])
def post():
    if request.method=='POST':
        name=request.form['name']
        return f"Hello {name} welcome Good morning"
    return render_template('form.html')

@app.route('/submit',methods=['GET','POST'])
def submit():
    if request.method=='POST':
        name=request.form['name']
        return f"Hello {name} welcome Good morning"
    return render_template('form.html')

@app.route('/success/<int:score>')
def succsess(score):
    print(f"you're score is {100-score}")
    mark=100-score
    if mark>50:
        res= "you are pass !"
    else:
        res="better luck next time"
    return render_template('result.html',result=res)

@app.route('/marksheet/<int:score>')
def marksheet(score):
    data={'hello':1}
    if score>50:
        data['Result']='Pass'
    else:
        data['Result']='Fail'
    data['Rank']='NA'
    data['percentage']=f'{(score/100)*100}%'
    data['mark']=score
    return render_template('marksheet.html',data=data)
@app.route('/passed/<float:total>/<float:average>')
def passed(average,total):
    return render_template('pass.html',average=average,total_score=total)

@app.route('/fail')
def Fail():
    return render_template('fail.html')

@app.route('/getresults')
def get_result():
    return render_template('getresult.html')


@app.route('/report_card',methods=['POST','GET'])
def report_card():
    if request.method=='POST':
        Tamil=float(request.form['Tamil'])#what ever we cature in form str
        English=float(request.form['English'])
        Maths=float(request.form['Maths'])
        Science=float(request.form['science'])
        Social=float(request.form['social'])
        total_score=Tamil+English+Maths+Science+Social
        average=total_score/5
    #     if (Tamil>50 and English>50 and Maths>50 and Science>50 and Social>50):
    #         return render_template('pass.html',total_score=total_score,average=average)
    #     else:
    #         return render_template('fail.html',total_score=total_score,average=average)
    # else:
    #     return render_template('getresult.html')
    if (Tamil>50 and English>50 and Maths>50 and Science>50 and Social>50): 
        return redirect(url_for('passed',total=total_score,average=average))
    

if __name__=="__main__":
    app.run(debug=True)