template <typename Real>
struct AsianOption
{
    enum CallPut {Call, Put};

    // Parameters
    Real spot;
    Real strike;
    Real r;
    Real sigma;
    Real tenor;
    Real dt;

    // Value
    Real golden;
    Real value;

    // Option type
    CallPut type;
};
