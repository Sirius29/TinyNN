#ifndef TINYNN_LAYER_H_
#define TINYNN_LAYER_H_

class Layer
{
public:
    Layer() {}
    ~Layer() {}
    virtual int forward() = 0;
    virtual int backward() = 0;

private:
};

class Dense : public Layer
{
public:
    Dense(){}
    ~Dense(){}
    int forward()
    {

    }

    int backward()
    {
        
    }
private:
};

#endif