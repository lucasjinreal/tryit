#include <iostream>
#include <map>
#include <vector>
#include <queue>


using namespace std;


int main() {

    map<int, vector<float>> samples;
    samples.insert({0, {0,3,4}});
    cout << samples.size() << endl;

    map<int, vector<float>>::iterator it = samples.find(1);
    cout << (it == samples.end()) << endl;

    int idx = 5376835;
    idx = idx % 9;
    cout << idx << endl;

    // define a certain size queue
    queue<int> aq;
    aq.push(9);
    aq.push(3);
    aq.push(1);

    while (!aq.empty())
    {
        cout << "front: " << aq.front() << " back: " << aq.back() << endl;
        aq.pop();
    }
    


}