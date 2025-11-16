from flask import Flask,jsonify,request
app=Flask(__name__)
items=[{'name':'jai' ,'age':24 ,'education':'BE'},
      {'name':'sai' ,'age':21 ,'education':'BE'} ]

@app.route('/items',methods=['GET'])
def get_items():
    return jsonify(items)

@app.route('/items/<name>',methods=['GET'])
def get_item(name):
    item=next((item for item in items if item["name"]==name),None)
    if item is None:
        return jsonify({'error':"Item not found"})
    return jsonify(item)
    
@app.route('/items/<name>',methods=['POST'])
def create_item():
    if not request.json or 'name' in request.json:
        return jsonify({'error':"Item not found"})
    new_item={'name':request.json['name'],'age':request.json['age'],'education':request.json['education']}
    items.append(new_item)
    return jsonify(new_item)

if __name__=='__main__':
    app.run()