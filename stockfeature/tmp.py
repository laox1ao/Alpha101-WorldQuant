import pickle
import pip, os, time

a = {1:1,2:2}
print a.update(dict([[3,3],[4,4]]))
print a 
exit()
for package in pip.get_installed_distributions():
         print "%s: %s" % (package, time.ctime(os.path.getctime(package.location)))

data = pickle.load(open('shang_index.pkl'))
print(len(data))
print(data.__class__)
print('\n'.join(map(str,data.items())))
