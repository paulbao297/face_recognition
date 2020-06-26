id_array= {"vu_1"}
arr=[]
dem=len(id_array)
print("count:",dem)
while dem<5:
	print("dem<5")
	inp=input()
	id_array.add(inp)
	#list(set(id_array))
	dem=len(id_array)
	print("id_arr:",id_array)
	print("count:",dem)
	for i in id_array:
		arr.append(i)
identity=arr[0]
identity=identity.split("_")[0]
print(identity)
