#include "model.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace tinynn;

int ReverseInt(int i)
{
	int ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return (ch1 << 24) | (ch2 << 16) | (ch3 << 8) | ch4;
}

int ReadImages(std::string &file_name, std::vector<Tensor> &set)
{
	std::ifstream ifile;
	ifile.open(file_name.c_str(), std::ios::in | std::ios::binary);
	if (!ifile.is_open())
	{
		std::cout << "open " << file_name << " fail." << std::endl;
		return -1;
	}

	int magic_num = 0;
	int number = 0, height = 0, width = 0;
	ifile.read((char *)&magic_num, sizeof(int));
	ifile.read((char *)&number, sizeof(int));
	ifile.read((char *)&height, sizeof(int));
	ifile.read((char *)&width, sizeof(int));
	magic_num = ReverseInt(magic_num);
	number = ReverseInt(number);
	height = ReverseInt(height);
	width = ReverseInt(width);
	std::cout << "magic_num is " << magic_num << std::endl;
	std::cout << "number of images is " << number << std::endl;
	std::cout << "height x width is " << height << " x " << width << std::endl;

	set.resize(number);
	char val = 0;
	int total = height * width;
	for (int i = 0; i < number; ++i)
	{
		// Tensor t(Size(height, width, 1), sizeof(char));
		// ifile.read(t.GetData<char>(), total * sizeof(char));
		// set[i] = std::move(t);

		Tensor t(Size(1, total, 1));
		float *p_data = t.GetData<float>();
		for (int j = 0; j < total; ++j)
		{
			ifile.read(&val, sizeof(char));
			p_data[j] = static_cast<float>(val);
		}
		set[i] = std::move(t);
	}

	ifile.close();

	return 0;
}

int ReadIabels(std::string &file_name, std::vector<Tensor> &set)
{
	std::ifstream ifile;
	ifile.open(file_name.c_str(), std::ios::in | std::ios::binary);
	if (!ifile.is_open())
	{
		std::cout << "open " << file_name << " fail." << std::endl;
		return -1;
	}
	int magic_num = 0, number = 0;
	ifile.read((char *)&magic_num, sizeof(int));
	ifile.read((char *)&number, sizeof(int));
	magic_num = ReverseInt(magic_num);
	number = ReverseInt(number);
	std::cout << "magic_num is " << magic_num << std::endl;
	std::cout << "number of images is " << number << std::endl;

	set.resize(number);
	char val = 0;
	int height = 1, width = 10;
	int total = height * width;
	for (int i = 0; i < number; ++i)
	{
		// Tensor t(Size(height, width, 1), sizeof(char));
		// ifile.read(t.GetData<char>(), total * sizeof(char));
		// set[i] = std::move(t);

		Tensor t(Size(1, total, 1));
		float *p_data = t.GetData<float>();
		for (int j = 0; j < total; ++j)
		{
			ifile.read(&val, sizeof(char));
			p_data[j] = static_cast<float>(val);
		}
		set[i] = std::move(t);
	}

	ifile.close();
	return 0;
}

int main()
{
	std::string train_img = "train-images.idx3-ubyte";
	std::string train_label = "train-labels.idx1-ubyte";
	std::string test_img = "t10k-images.idx3-ubyte";
	std::string test_label = "t10k-labels.idx1-ubyte";

	std::vector<Tensor> train_set, test_set;
	std::vector<Tensor> train_target, test_target;
	if (ReadImages(train_img, train_set) != 0)
	{
		std::cout << "ReadImages fail." << std::endl;
		return -1;
	}

	if (ReadImages(test_img, test_set) != 0)
	{
		std::cout << "ReadImages fail." << std::endl;
		return -1;
	}

	if (ReadIabels(train_label, train_target) != 0)
	{
		std::cout << "ReadIabels fail." << std::endl;
		return -1;
	}

	if (ReadIabels(test_label, test_target) != 0)
	{
		std::cout << "ReadIabels fail." << std::endl;
		return -1;
	}

	std::unique_ptr<Initializer> p_normal = std::make_unique<Normal>();
	std::unique_ptr<Initializer> p_zero = std::make_unique<Zeros>();

	Model model;
	model.loss_func = std::make_unique<MSE>();
	model.optim = std::make_unique<SGD>();
	model.net.layers.emplace_back(std::make_unique<Dense>(200, p_normal.get(), p_zero.get()));
	// model.net.layers.emplace_back(std::make_unique<ReLU>());
	// model.net.layers.emplace_back(std::make_unique<Dense>(100, p_normal.get(), p_zero.get()));
	// model.net.layers.emplace_back(std::make_unique<ReLU>());
	// model.net.layers.emplace_back(std::make_unique<Dense>(70, p_normal.get(), p_zero.get()));
	// model.net.layers.emplace_back(std::make_unique<ReLU>());
	// model.net.layers.emplace_back(std::make_unique<Dense>(30, p_normal.get(), p_zero.get()));
	// model.net.layers.emplace_back(std::make_unique<ReLU>());
	// model.net.layers.emplace_back(std::make_unique<Dense>(10, p_normal.get(), p_zero.get()));

	Tensor out = model.Forward(train_set[0]);
	std::cout << out.GetSize() << std::endl;
	// for (int i = 0; i <)

	return 0;
}