#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <unordered_set>
using namespace std;
double quant(vector<double>&v,double p){double k=(v.size()-1)*p;size_t j=(size_t)k;return v[j]+(v[min(j+1,v.size()-1)]-v[j])*(k-j);}
int main(int argc,char**argv){
 if(argc!=4&&argc!=6)return 2;int n=stoi(argv[1]),bp=stoi(argv[2]);ofstream out(argv[3]);out<<setprecision(12);
 unordered_set<string> selected;ofstream points;bool extracting=argc==6;if(extracting){ifstream filter(argv[4]);string line;while(getline(filter,line))selected.insert(line);points.open(argv[5]);points<<setprecision(17)<<"time,column_index,component,value\n";}
 out<<"hour,column_index,component,n,median,q25,q75,mean,std,zero,repeated,first_time,last_time\n";
 vector<vector<double>> vals(n*(bp?2:1));vector<long long> zeros(vals.size()),reps(vals.size());vector<double> prev(vals.size(),NAN);vector<string> first(vals.size()),last(vals.size());
 string hour;long long rows=0,obs=0,bad=0,reverse=0,duplicates=0;string previous;
 auto flush=[&](){if(hour.empty())return;for(size_t j=0;j<vals.size();j++){auto&v=vals[j];if(v.empty())continue;sort(v.begin(),v.end());double mean=0,var=0;for(double x:v)mean+=x;mean/=v.size();for(double x:v)var+=(x-mean)*(x-mean);
 out<<hour<<":00:00,"<<(bp?j/2:j)<<","<<(bp?j%2:0)<<","<<v.size()<<","<<quant(v,.5)<<","<<quant(v,.25)<<","<<quant(v,.75)<<","<<mean<<","<<(v.size()>1?sqrt(var/(v.size()-1)):0)<<","<<zeros[j]<<","<<reps[j]<<","<<first[j]<<","<<last[j]<<"\n";v.clear();zeros[j]=reps[j]=0;first[j]=last[j]="";}};
 auto number=[](const string&s,double&v){char*e;v=strtod(s.c_str(),&e);while(*e==' '||*e=='\r'||*e=='\t')e++;return e!=s.c_str()&&!*e&&isfinite(v);};
 string field,stamp;int col=0;bool quoted=false,afterquote=false,header=true;
 auto finishfield=[&](){if(header){field.clear();return;}if(col==0){stamp=field;if(stamp.size()>10&&stamp[10]=='T')stamp[10]=' ';for(size_t k=11;k<stamp.size();k++)if(stamp[k]=='-')stamp[k]=':';for(size_t k=0;k<10&&k<stamp.size();k++)if(stamp[k]=='/')stamp[k]='-';int y=0,m=0,d=0,h=0,mi=0;double sec=0;int parsed=sscanf(stamp.c_str(),"%d-%d-%d %d:%d:%lf",&y,&m,&d,&h,&mi,&sec);if(parsed<5||y<2000||m<1||m>12||d<1||d>31||h<0||h>23||mi<0||mi>59||sec<0||sec>=60){bad++;stamp="";}else{char canonical[40];snprintf(canonical,sizeof(canonical),"%04d-%02d-%02d %02d:%02d:%02d",y,m,d,h,mi,(int)sec);stamp=canonical;if(!previous.empty()&&stamp<previous)reverse++;if(stamp==previous)duplicates++;previous=stamp;string hr=stamp.substr(0,13);if(hr!=hour){flush();hour=hr;}}}else if(!stamp.empty()&&col<=n&&(!extracting||selected.count(stamp.substr(0,13)+"|"+to_string(col-1)))){double x,y;bool ok=false;if(bp){size_t q=field.find('/');if(q!=string::npos)ok=number(field.substr(0,q),x)&&number(field.substr(q+1),y);}else ok=number(field,x);if(ok){for(int c=0;c<(bp?2:1);c++){double a=c?y:x;size_t j=(col-1)*(bp?2:1)+c;if(extracting){points<<stamp<<","<<col-1<<","<<c<<","<<a<<"\n";continue;}vals[j].push_back(a);zeros[j]+=(a==0);reps[j]+=isfinite(prev[j])&&a==prev[j];prev[j]=a;if(first[j].empty())first[j]=stamp;last[j]=stamp;}obs++;}}field.clear();};
 char ch;while(cin.get(ch)){if(afterquote){afterquote=false;if(ch=='"'){field.push_back(ch);continue;}quoted=false;}if(quoted){if(ch=='"')afterquote=true;else field.push_back(ch);continue;}if(ch=='"'&&field.empty()){quoted=true;continue;}if(ch==','){finishfield();col++;continue;}if(ch=='\n'){finishfield();if(header)header=false;else rows++;col=0;continue;}if(ch!='\r')field.push_back(ch);}
 if(!field.empty()||col){finishfield();if(!header)rows++;}flush();out.close();
 cerr<<"{\"rows\":"<<rows<<",\"observed\":"<<obs<<",\"bad_time_rows\":"<<bad<<",\"reverse_time_rows\":"<<reverse<<",\"duplicate_time_rows\":"<<duplicates<<"}"<<endl;return bad||reverse||duplicates?3:0;
}
