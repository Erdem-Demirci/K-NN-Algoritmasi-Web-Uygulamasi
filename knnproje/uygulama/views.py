from django.http import HttpResponse
from django.shortcuts import render
from .forms import UploadFileForm
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

def types(dosya_yolu):
    file_path = dosya_yolu
    data = pd.read_csv(file_path)
    kolon = data.columns[:].tolist()
    degerler = [(kol, str(data[kol].dtype)) for kol in kolon]
    return degerler

def types2(dosya_yolu, nitelikler):
    file_path = dosya_yolu
    data = pd.read_csv(file_path)
    degerler2 = [(nitelik, str(data[nitelik].dtype)) for nitelik in nitelikler]
    return degerler2

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            if file.name.endswith('.csv'):
                dosya_yolu = dosyayi_kaydet(file)
                request.session['dosya_yolu'] = dosya_yolu
                kolon = islemler(dosya_yolu)
                degerler = types(dosya_yolu)
                request.session['kolon'] = kolon
                request.session['degerler'] = degerler
                data = {
                    "kolon": kolon,
                    "degerler": degerler
                }
                return render(request, 'secim.html', data)
            else:
                form = UploadFileForm()
                return render(request, 'administration/upload.html', {'form': form, 'error_message': 'Lütfen CSV dosyası yükleyiniz.'})
    else:
        form = UploadFileForm()
    return render(request, 'administration/upload.html', {'form': form})

def dosyayi_kaydet(file):
    hedef_dizin = "dosyalar"
    kaydedilen_yol = hedef_dizin + '/' + file.name
    
    with open(kaydedilen_yol, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    return kaydedilen_yol

def islemler(dosya_yolu):
    file_path = dosya_yolu
    data = pd.read_csv(file_path)
    kolon = data.columns[:].tolist()
    return kolon

def secim(request):
    if request.method == 'POST':
        nitelikler = request.POST.getlist('nitelikler')
        request.session['nitelikler'] = nitelikler
        dosya_yolu = request.session.get('dosya_yolu')
        degerler = types2(dosya_yolu, nitelikler)
        request.session['degerler2'] = degerler
        sinif = request.POST.get('sinif')
        min_k = request.POST.get('min_k')
        max_k = request.POST.get('max_k')
        request.session['sinif'] = sinif
        if not nitelikler or not sinif or not min_k or not max_k:
            kolon = request.session.get('kolon')
            degerler = request.session.get('degerler')
            data = {
                'kolon': kolon,
                'degerler': degerler,
                'error_message' : 'Lutfen tum alanlari doldurun'
            }
            return render(request, 'secim.html', data)
        min_k = int(min_k)
        max_k  = int(max_k)

        if dosya_yolu:
            data = pd.read_csv(dosya_yolu)

            for kol, dtype in degerler:
                if dtype == 'object':
                    mode_value = data[kol].mode()[0]
                    data[kol].fillna(mode_value, inplace=True)
                    le = LabelEncoder()
                    data[kol] = le.fit_transform(data[kol])

            numerical_columns = data.select_dtypes(include=np.number).columns
            for column in numerical_columns:
                mean_value = data[column].mean()
                data[column].fillna(mean_value, inplace=True)

            X = data[nitelikler].values
            y = data[sinif].values

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=0)

        dizi = list(range(min_k,max_k+1))
        
        grid_params = { 'n_neighbors' : dizi,
               'weights' : ['uniform','distance'],
               'metric' : ['euclidean','manhattan']}
        
        gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose = 1, cv=3, n_jobs = -1)

        g_res = gs.fit(X_train, y_train)

        best_k = g_res.best_params_['n_neighbors']
        best_metric = g_res.best_params_['metric']

        data = {
            'degerler': degerler,
            'nitelikler': nitelikler,
            'best_k': best_k,
            'best_metric' : best_metric
        }
        request.session['best_k'] = best_k
        request.session['best_metric'] = best_metric


        return render(request, 'arayuz.html', data)
    else:
        kolon = request.session.get('kolon')
        degerler = request.session.get('degerler')
        data = {
            'kolon': kolon,
            'degerler': degerler
        }
        return render(request, 'secim.html', data)

def arayuz(request):
    if request.method == 'POST':
        elemanlar = request.POST.getlist('deger[]')
        k_degeri = int(request.POST.get('k_degeri'))
        secim = int(request.POST.get('secim'))

        dosya_yolu = request.session.get('dosya_yolu')
        nitelikler = request.session.get('nitelikler')
        sinif = request.session.get('sinif')
        degerler = request.session.get('degerler2')
        best_k = request.session.get('best_k')
        best_metric = request.session.get('best_metric')

        if dosya_yolu:
            data = pd.read_csv(dosya_yolu)

            for kol, dtype in degerler:
                if dtype == 'object':
                    mode_value = data[kol].mode()[0]
                    data[kol].fillna(mode_value, inplace=True)
                    le = LabelEncoder()
                    data[kol] = le.fit_transform(data[kol])

            numerical_columns = data.select_dtypes(include=np.number).columns
            for column in numerical_columns:
                mean_value = data[column].mean()
                data[column].fillna(mean_value, inplace=True)

            X = data[nitelikler].values
            y = data[sinif].values

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=0)

            if secim == 1:
                knn = KNeighborsClassifier(n_neighbors=k_degeri, metric='euclidean')
            elif secim == 2:
                knn = KNeighborsClassifier(n_neighbors=k_degeri, metric='manhattan')
            elif secim == 3:
                knn = KNeighborsClassifier(n_neighbors=k_degeri, metric='minkowski', p=2)
            else:
                knn = KNeighborsClassifier(n_neighbors=k_degeri, metric='minkowski', p=1)

            knn.fit(X_train, y_train)

            pred = knn.predict(X_test)

            matris = confusion_matrix(y_test, pred)
            report = classification_report(y_test, pred)

            elemanlar_encoded = []
            for i, (kol, dtype) in enumerate(degerler):
                if dtype == 'int64' or dtype == 'float64':
                    elemanlar_encoded.append(float(elemanlar[i]))
                elif dtype == 'object':
                    le = LabelEncoder()
                    elemanlar_encoded.append(le.fit_transform([elemanlar[i]])[0])
                else:
                    elemanlar_encoded.append(elemanlar[i])

            elemanlar_x = np.array(elemanlar_encoded)
            elemanlar_x_reshaped = elemanlar_x.reshape(1, -1)

            scaled_x = scaler.transform(elemanlar_x_reshaped)

            tahmin = knn.predict(scaled_x)
            tahmin = tahmin[0]

            knn_train_accuracy = knn.score(X_train, y_train)
            knn_test_accuracy = knn.score(X_test, y_test)

             # Confusion matrisini string olarak biçimlendir
            matris_str = np.array2string(matris, separator=', ')

            sonuclar = {
                'degerler': degerler,
                'nitelikler': nitelikler,
                'matris': matris_str,
                'report': report,
                'tahmin': tahmin,
                'knn_train_accuracy': knn_train_accuracy,
                'knn_test_accuracy': knn_test_accuracy,
                'best_k' : best_k,
                'best_metric' : best_metric
            }
            return render(request, 'arayuz.html', sonuclar)
        else:
            return HttpResponse("Dosya yolu bulunamadı.")
    else:
        kolon = request.session.get('kolon')
        degerler = request.session.get('degerler')
        data = {
            'kolon': kolon,
            'degerler': degerler
        }
        return render(request, 'secim.html', data)
