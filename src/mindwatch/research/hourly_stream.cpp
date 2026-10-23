#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {
constexpr std::size_t max_columns = 100000;
constexpr std::size_t max_field_bytes = 1024 * 1024;
constexpr std::size_t max_record_bytes = 64 * 1024 * 1024;
std::string trim(const std::string& s) {
    const auto begin = s.find_first_not_of(" \t\r\n");
    return begin == std::string::npos ? "" : s.substr(begin, s.find_last_not_of(" \t\r\n") - begin + 1);
}
// Bounded RFC-style fields; malformed quoting never leaks into later rows.
bool read_record(std::istream& input, std::vector<std::string>& fields) {
    enum class State { plain, quoted, closed };
    State state = State::plain;
    std::string field;
    fields.clear();
    std::size_t bytes = 0;
    bool any = false;
    char ch;
    auto finish = [&]() {
        fields.push_back(field); field.clear();
        if (fields.size() > max_columns + 1) throw std::runtime_error("column_limit");
    };
    while (input.get(ch)) {
        any = true;
        if (++bytes > max_record_bytes) throw std::runtime_error("record_limit");
        if (ch == '\0') throw std::runtime_error("nul_byte");
        if (state == State::quoted) {
            if (ch == '"') state = State::closed;
            else field.push_back(ch);
        } else if (state == State::closed && ch == '"') {
            field.push_back('"'); state = State::quoted;
        } else if (ch == ',' || ch == '\n' || ch == '\r') {
            finish(); state = State::plain;
            if (ch != ',') {
                if (ch == '\r' && input.peek() == '\n') input.get();
                return true;
            }
        } else if (state == State::closed) throw std::runtime_error("characters_after_quote");
        else if (ch == '"') {
            if (!field.empty()) throw std::runtime_error("quote_inside_plain_field");
            state = State::quoted;
        } else field.push_back(ch);
        if (field.size() > max_field_bytes) throw std::runtime_error("field_limit");
    }
    if (!input.eof()) throw std::runtime_error("input_read_failure");
    if (state == State::quoted) throw std::runtime_error("unterminated_quote");
    if (!any) return false;
    finish(); return true;
}
bool missing(const std::string& s) {
    std::string lower = trim(s);
    for (char& ch : lower) if (ch >= 'A' && ch <= 'Z') ch += 'a' - 'A';
    return lower.empty() || lower == "nan" || lower == "na" || lower == "n/a" || lower == "null" || lower == "none";
}
double number(const std::string& raw) {
    const std::string s = trim(raw);
    if (s.size() > 128) throw std::runtime_error("numeric_field_limit");
    static const std::regex syntax(R"([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)");
    if (!std::regex_match(s, syntax)) throw std::runtime_error("invalid_numeric_field");
    char* end = nullptr; errno = 0;
    const double value = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || *end || errno == ERANGE || !std::isfinite(value)) throw std::runtime_error("nonfinite_or_out_of_range_number");
    return value;
}
struct Stamp { std::string display, order, hour; };
Stamp timestamp(const std::string& raw) {
    // Local wall-clock export only: offsets require prior explicit resolution.
    static const std::regex syntax(R"(([0-9]{4})[-/.]([0-9]{1,2})[-/.]([0-9]{1,2})[ T]([0-9]{1,2})[:-]([0-9]{2})(?:[:-]([0-9]{2})(?:\.([0-9]{1,9}))?)?)");
    std::smatch match; const std::string text = trim(raw);
    if (text.size() > 64) throw std::runtime_error("timestamp_field_limit");
    if (!std::regex_match(text, match, syntax)) throw std::runtime_error("invalid_local_timestamp");
    const int y=std::stoi(match[1]), m=std::stoi(match[2]), d=std::stoi(match[3]);
    const int h=std::stoi(match[4]), minute=std::stoi(match[5]), second=match[6].matched?std::stoi(match[6]):0;
    const int month_days[]={31,28,31,30,31,30,31,31,30,31,30,31};
    const bool leap=y%4==0 && (y%100!=0 || y%400==0);
    if (y<2000 || m<1 || m>12 || d<1 || d>month_days[m-1]+(m==2 && leap?1:0) || h>23 || minute>59 || second>59)
        throw std::runtime_error("invalid_calendar_timestamp");
    char canonical[32];
    std::snprintf(canonical,sizeof canonical,"%04d-%02d-%02d %02d:%02d:%02d",y,m,d,h,minute,second);
    std::string fraction=match[7].matched?match[7].str():"";
    const std::string order=std::string(canonical)+"."+fraction+std::string(9-fraction.size(),'0');
    while (!fraction.empty() && fraction.back()=='0') fraction.pop_back();
    return {std::string(canonical)+(fraction.empty()?"":"."+fraction),order,std::string(canonical).substr(0,13)};
}
double quantile(const std::vector<double>& v,double p) {
    const double k=(v.size()-1)*p; const auto j=static_cast<std::size_t>(k);
    return static_cast<double>((1-(k-j))*static_cast<long double>(v[j])+(k-j)*static_cast<long double>(v[std::min(j+1,v.size()-1)]));
}
}
int main(int argc,char** argv) {
    try {
        if (argc!=4 && argc!=6) throw std::runtime_error("argument_count");
        const std::string count_arg=argv[1],bp_arg=argv[2];
        if (count_arg.empty() || count_arg.find_first_not_of("0123456789")!=std::string::npos) throw std::runtime_error("invalid_column_count");
        const auto n=std::stoull(count_arg);
        if (n<1 || n>max_columns || (bp_arg!="0" && bp_arg!="1")) throw std::runtime_error("invalid_dimensions");
        const bool bp=bp_arg=="1",extracting=argc==6;
        std::ofstream out(argv[3]);
        if (!out) throw std::runtime_error("output_open_failure");
        out<<std::setprecision(12);
        std::unordered_set<std::string> selected; std::ofstream points;
        if (extracting) {
            std::ifstream filter(argv[4]);
            if (!filter) throw std::runtime_error("filter_open_failure");
            std::string line; char ch;
            auto add_filter=[&]() {
                if (selected.size()>=1000000) throw std::runtime_error("filter_limit");
                selected.insert(trim(line)); line.clear();
            };
            while (filter.get(ch)) {
                if (ch=='\n') add_filter();
                else {if (line.size()>=128) throw std::runtime_error("filter_limit");line.push_back(ch);}
            }
            if (!filter.eof()) throw std::runtime_error("filter_read_failure");
            if (!line.empty()) add_filter();
            points.open(argv[5]); if (!points) throw std::runtime_error("points_open_failure");
            points<<std::setprecision(17)<<"time,column_index,component,value\n";
        }
        out<<"hour,column_index,component,n,median,q25,q75,mean,std,zero,repeated,first_time,last_time\n";
        const std::size_t components=bp?2:1;
        std::vector<std::vector<double>> values(n*components);
        std::vector<unsigned long long> zeros(values.size()),repeats(values.size());
        std::vector<double> previous_value(values.size(),std::numeric_limits<double>::quiet_NaN());
        std::vector<std::string> first(values.size()),last(values.size());
        std::string hour,previous_time; unsigned long long rows=0,observations=0,retained=0;
        auto flush=[&]() {
            if (hour.empty()) return;
            for (std::size_t j=0;j<values.size();++j) {
                auto& v=values[j]; if (v.empty()) continue;
                std::sort(v.begin(),v.end()); long double mean=0,variance=0;
                for (double value:v) mean+=static_cast<long double>(value)/v.size();
                for (double value:v) variance+=(value-mean)*(value-mean)/v.size();
                const double sd=v.size()>1?static_cast<double>(std::sqrt(variance*v.size()/(v.size()-1))):0;
                if (!std::isfinite(static_cast<double>(mean)) || !std::isfinite(sd)) throw std::runtime_error("aggregate_out_of_range");
                out<<hour<<":00:00,"<<j/components<<','<<j%components<<','<<v.size()<<','<<quantile(v,.5)<<','<<quantile(v,.25)<<','<<quantile(v,.75)<<','<<static_cast<double>(mean)<<','<<sd<<','<<zeros[j]<<','<<repeats[j]<<','<<first[j]<<','<<last[j]<<'\n';
                std::vector<double>().swap(v); zeros[j]=repeats[j]=0; first[j].clear(); last[j].clear();
            }
            if (!out) throw std::runtime_error("output_write_failure");
            retained=0;
        };
        std::vector<std::string> fields;
        if (!read_record(std::cin,fields) || fields.size()!=n+1) throw std::runtime_error("missing_or_invalid_header");
        if (fields[0].compare(0,3,"\xef\xbb\xbf")==0) fields[0].erase(0,3);
        std::string header=trim(fields[0]);
        std::transform(header.begin(),header.end(),header.begin(),[](unsigned char c){return std::tolower(c);});
        if (header!="time" && header!="timestamp" && header!="date" && header!="regdt" && header!="-") throw std::runtime_error("unrecognized_time_header");
        std::unordered_set<std::string> headers;
        for (std::size_t j=1;j<fields.size();++j)
            if (trim(fields[j]).empty() || !headers.insert(trim(fields[j])).second) throw std::runtime_error("duplicate_or_empty_header");
        while (read_record(std::cin,fields)) {
            if (fields.size()!=n+1) throw std::runtime_error("row_column_count");
            const Stamp stamp=timestamp(fields[0]);
            if (!previous_time.empty() && stamp.order<=previous_time) throw std::runtime_error("duplicate_or_reverse_time");
            previous_time=stamp.order;
            if (stamp.hour!=hour) {flush();hour=stamp.hour;}
            ++rows;
            for (std::size_t column=0;column<n;++column) {
                const std::string field=trim(fields[column+1]); if (missing(field)) continue;
                double pair[2];
                if (bp) {
                    const auto slash=field.find('/');
                    if (slash==std::string::npos || field.find('/',slash+1)!=std::string::npos) throw std::runtime_error("invalid_pressure_pair");
                    pair[0]=number(field.substr(0,slash));pair[1]=number(field.substr(slash+1));
                } else pair[0]=number(field);
                if (extracting && !selected.count(stamp.hour+"|"+std::to_string(column))) continue;
                for (std::size_t c=0;c<components;++c) {
                    const double value=pair[c];
                    if (extracting) {points<<stamp.display<<','<<column<<','<<c<<','<<value<<'\n';continue;}
                    const std::size_t j=column*components+c;
                    if (values[j].size()>=1000000 || ++retained>10000000) throw std::runtime_error("hour_observation_limit");
                    values[j].push_back(value); zeros[j]+=value==0;
                    repeats[j]+=std::isfinite(previous_value[j]) && value==previous_value[j];previous_value[j]=value;
                    if (first[j].empty()) first[j]=stamp.display;
                    last[j]=stamp.display;
                }
                ++observations;
            }
        }
        flush();out.close();if (!out) throw std::runtime_error("output_close_failure");
        if (extracting) {points.close();if (!points) throw std::runtime_error("points_write_failure");}
        std::cerr<<"{\"rows\":"<<rows<<",\"observed\":"<<observations<<",\"bad_time_rows\":0,\"reverse_time_rows\":0,\"duplicate_time_rows\":0}"<<std::endl;
        return 0;
    } catch (const std::exception&) {
        // Never echo input fields, paths, participant headers or untrusted argv.
        std::cerr<<"hourly_parser_invalid_input_or_io_failure"<<std::endl;
        return 3;
    }
}
