#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <omp.h>

using namespace std;

// Calcola la distanza Euclidea tra due punti
double distance(vector<double>& a, vector<double>& b) {
    double dist = 0.0;
    for (int i = 0; i < a.size(); i++) {
        dist += pow(a[i] - b[i], 2);
    }
    return sqrt(dist);
}

// Esegue l'algoritmo K-means clustering in parallelo
void kMeansClustering(vector<vector<double>>& data, int k, vector<vector<double>>& centroids, vector<int>& clusters) {
    int n = data.size();
    int dim = data[0].size();
    
    // Inizializza i centroidi in modo casuale
    #pragma omp parallel for
    for (int i = 0; i < k; i++) {
        int idx = rand() % n;
        centroids[i] = data[idx];
    }
    
    // Assegna ogni punto al cluster più vicino
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        double minDist = INFINITY;
        int minIdx = 0;
        for (int j = 0; j < k; j++) {
            double dist = distance(data[i], centroids[j]);
            if (dist < minDist) {
                minDist = dist;
                minIdx = j;
            }
        }
        clusters[i] = minIdx;
    }
    
    // Ripete fino alla convergenza
    bool changed = true;
    while (changed) {
        // Calcola i nuovi centroidi
        vector<int> counts(k, 0);
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            int cluster = clusters[i];
            #pragma omp atomic
            counts[cluster]++;
            #pragma omp parallel for
            for (int j = 0; j < dim; j++) {
                #pragma omp atomic
                centroids[cluster][j] += data[i][j];
            }
        }
        #pragma omp parallel for
        for (int i = 0; i < k; i++) {
            if (counts[i] > 0) {
                #pragma omp parallel for
                for (int j = 0; j < dim; j++) {
                    centroids[i][j] /= counts[i];
                }
            }
        }
        
        // Assegna ogni punto al cluster più vicino
        changed = false;
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            double minDist = INFINITY;
            int minIdx = 0;
            for (int j = 0; j < k; j++) {
                double dist = distance(data[i], centroids[j]);
                if (dist < minDist) {
                    minDist = dist;
                    minIdx = j;
                }
            }
            if (minIdx != clusters[i]) {
                #pragma omp critical
                {
                    changed = true;
                    clusters[i] = minIdx;
                }
            }
        }
    }
}

int main() {
    // Genera alcuni dati casuali
    vector<vector<double>> data = {
    {-3.54769, -4.04248},
	{-5.54063, 2.93299},
	{-0.222459, -1.8501},
	{-5.75519, -6.43216},
	{-3.51682, 0.696472},
	{-5.53426, -6.85545},
	{1.12418, 5.35785},
	{-8.3267, 7.05604},
	{1.70881, -3.31232},
	{-7.76195, 5.70419},
	{8.99625, -4.34928},
	{-7.88853, -8.05076},
	{-5.5348, -3.84489},
	{8.52193, 3.29888},
	{-8.73137, -4.76812},
	{-1.20446, -0.489068},
	{-8.57576, -7.65058},
	{1.83471, -7.15798},
	{-2.70913, -1.30594},
	{-4.32888, -0.711923},
	{1.24529, 8.9211},
	{1.28114, 3.08364},
	{4.21009, -1.07775},
	{-9.01683, 7.49712},
	{-7.79474, -5.43359},
	{5.77238, -6.79855},
	{7.80014, 4.98107},
	{0.0771777, 1.10719},
	{-9.78733, 7.32934},
	{5.56639, 5.6667},
	{-7.33402, -4.23243},
	{3.85867, -5.18741},
	{8.74271, -8.65787},
	{-8.7869, -5.21691},
	{3.8001, -4.12688},
	{4.92108, 7.5785},
	{-7.43403, 0.394626},
	{-0.815484, 8.74028},
	{9.58205, 5.44067},
	{-7.49673, -5.15685},
	{0.189537, -3.74104},
	{-0.868941, 7.55281},
	{3.69774, 9.61719},
	{-5.00516, -3.29126},
	{6.53601, -9.37459},
	{8.54687, -1.04503},
	{-9.25153, -4.61775},
	{6.61791, -1.92224},
	{-6.68363, -2.86065},
	{0.871443, 4.54467},
	{-2.97406, 2.11048},
	{7.58331, 8.95583},
	{3.46649, -6.42185},
	{-7.11852, 4.02722},
	{7.69821, -7.12361},
	{-3.30084, 2.17168},
	{-0.957935, 9.08023},
	{4.47371, 2.65082},
	{7.64206, -7.09144},
	{-7.24132, 7.32649},
	{-5.21185, 2.32083},
	{-3.95894, -9.34037},
	{-0.481434, -3.8364},
	{0.814806, 3.18339},
	{-0.0459847, 8.5021},
	{-9.98107, -3.45437},
	{7.48494, 5.17495},
	{-6.98164, 0.172526},
	{9.80023, 6.47337},
	{1.26964, -3.9466},
	{-6.31553, -3.08858},
	{-7.86723, 5.64766},
	{-1.6091, 5.35087},
	{9.49158, 1.5569},
	{1.42939, -6.91507},
	{-0.941234, -2.62294},
	{-8.7148, -1.76153},
	{-7.35539, -8.41082},
	{-6.87137, -8.27035},
	{1.59285, 3.10641},
	{3.31816, 5.96561},
	{-1.48326, 5.59252},
	{5.75567, 6.7596},
	{-8.10331, 1.78253},
	{-6.48217, -5.73985},
	{-6.59794, -2.12347},
	{0.88151, -9.49881},
	{-4.53656, 2.28354},
	{-9.16852, -8.59895},
	{-7.76827, 0.488911},
	{-1.6855, 3.00175},
	{-8.6144, -2.32497},
	{-0.38059, 0.141184},
	{-9.35231, -6.23062},
	{-6.71757, -7.2753},
	{-0.678574, 8.22237},
	{6.54649, -9.46854},
	{-8.9496, 9.04184},
	{-8.06817, 8.69395},
	{-6.02963, -6.51254},
	{-7.3259, 9.14411}
    };
    
    // Numero di cluster
    int k = 2; 
	// Centroidi
    vector<vector<double>> centroids(k, vector<double>(2));
    // Cluster	
    vector<int> clusters(data.size()); 
    
	// Misura il tempo di elaborazione
    clock_t start = clock();
	
    // Esegue l'algoritmo K-means clustering in parallelo
    kMeansClustering(data, k, centroids, clusters);
	
	// Calcola il tempo di elaborazione
    clock_t end = clock();
    double elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
    
    // Stampa i risultati
    cout << "Centroidi:" << endl;
    for (int i = 0; i < k; i++) {
        cout << "(" << centroids[i][0] << ", " << centroids[i][1] << ")" << endl;
    }
    cout << "Cluster:" << endl;
    for (int i = 0; i < data.size(); i++) {
        cout << "(" << data[i][0] << ", " << data[i][1] << ") -> " << clusters[i] << endl;
    }
    cout << "Tempo di elaborazione parallelo: " << elapsed_secs << " secondi" << endl;
	 
    return 0;
}
