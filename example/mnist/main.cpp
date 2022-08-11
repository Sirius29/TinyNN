#include "tensor.hpp"

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
	// set = std::vector<Tensor>(number, {height, width, 1, sizeof(char)});
	std::cout << set.capacity() << std::endl;
	char val = 0;
	int total = height * width;
	for (int i = 0; i < number; ++i)
	{
		Tensor t(Size(height, width, 1), sizeof(char));
		ifile.read(t.GetData<char>(), total * sizeof(char));
		set[i] = std::move(t);
	}

	ifile.close();

	return 0;
}

int ReadIabels(std::string &file_name)
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

	ifile.close();
	return 0;
}

int main()
{
	std::string train_img = "train-images.idx3-ubyte";
	std::string train_lab = "train-labels.idx1-ubyte";
	std::string test_img = "t10k-images.idx3-ubyte";
	std::string test_lab = "t10k-labels.idx1-ubyte";

	std::vector<Tensor> train_set, test_set;
	// std::vector<int>
	if (ReadImages(train_img, train_set) != 0)
	{
		std::cout << "ReadImages fail." << std::endl;
		return -1;
	}

	unsigned char *p = train_set[5].GetData<unsigned char>();
	for (int i = 0; i < 28; ++i)
	{
		for (int j = 0; j < 28; ++j)
		{
			int idx = i * 28 + j;
			printf("%4d ", p[idx]);
		}
		std::cout << std::endl;
	}

	return 0;
}