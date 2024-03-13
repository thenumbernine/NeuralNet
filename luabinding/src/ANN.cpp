// here's me trying to make c++ automated Lua binding
#include "NeuralNet/ANN.h"
#include <lua.hpp>
#include <type_traits>

void * operator new(size_t size, lua_State * L) {
	return lua_newuserdata(L, size);
}

template<typename T>
struct Info;

template<typename T>
struct InfoBase {
	static void mtinit(lua_State * L) {
		if (luaL_newmetatable(L, Info<T>::mtname)) {
			luaL_setfuncs(L, Info<T>::mtfields, 0);
		}
		lua_pop(L, 1);
	
		for (auto & pair : Info<T>::fields) {
			pair.second->mtinit(L);
		}
	}

	static int __index(lua_State * L, T * o) {
		char const * const k = lua_tostring(L, 2);
		auto const & fields = Info<T>::fields;
		auto iter = fields.find(k);
		if (iter == fields.end()) {
			lua_pushnil(L);
			return 1;
		}
		iter->second->push(*o, L);
		return 1;
	}

	static int __newindex(lua_State * L, T * o) {
		char const * const k = lua_tostring(L, 2);
		auto const & fields = Info<T>::fields;
		auto iter = fields.find(k);
		if (iter == fields.end()) {
			luaL_error(L, "sorry, this table cannot accept new members");
		}
		iter->second->read(*o, L, 3);
		return 0;
	}
};

template<typename T>
struct Info : public InfoBase<T> {};

template<typename T>
static T * lua_getptr(lua_State * L, int index) {
	luaL_checktype(L, index, LUA_TTABLE);
	lua_getfield(L, index, "ptr");
	luaL_checktype(L, -1, LUA_TUSERDATA);
	// verify metatable is Info<T>::mtname ... however lua_checkudata() does this but only for non-light userdata ...
	T * o = (T*)lua_touserdata(L, -1);
	lua_pop(L, 1);
	if (!o) luaL_error(L, "tried to index a null pointer");
	return o;
}

template<typename T>
static inline int __index(lua_State * L) {
	return Info<T>::__index(L, lua_getptr<T>(L, 1));
}

template<typename T>
static inline int __newindex(lua_State * L) {
	return Info<T>::__newindex(L, lua_getptr<T>(L, 1));
}

template<typename T>
static inline int __len(lua_State * L) {
	return Info<T>::__len(L, lua_getptr<T>(L, 1));
}

template<typename T>
static inline int __call(lua_State * L) {
	return Info<T>::__call(L, lua_getptr<T>(L, 1));
}



// general case just wraps the memory
template<typename T>
struct LuaRW {
	static void push(lua_State * L, T & v) {
		// hmm, we want a metatable for what we return
		// but lightuserdata has no metatable
		// so it'll have to be a new lua table that points back to this
		lua_newtable(L);
		luaL_setmetatable(L, Info<T>::mtname);
		lua_pushlightuserdata(L, &v);
		lua_setfield(L, -2, "ptr");
	}

	static T read(lua_State * L, int index) {
		// TODO how about assigning-by-value C++ objs to tables / deserialization from Lua?
		luaL_error(L, "this field cannot be overwritten");
		return {};	// this is cae for return-by-out-arg
	}
};


template<typename T>
requires (std::is_floating_point_v<T>)
struct LuaRW<T> {
	static void push(lua_State * L, T v) {
		lua_pushnumber(L, v);
	}
	static T read(lua_State * L, int index) {
		return lua_tonumber(L, index);
	}
};

template<typename T>
requires (std::is_integral_v<T>)
struct LuaRW<T> {
	static void push(lua_State * L, T v) {
		lua_pushinteger(L, v);
	}

	static T read(lua_State * L, int index) {
		return lua_tointeger(L, index);
	}
};

template<typename Base>
struct FieldBase {
	virtual ~FieldBase() {};
	// o can't be const because o.*field can't be const because in case it's a blob/ptr then it's getting pushed into lua as lightuserdata ... and can't be const
	virtual void push(Base & o, lua_State * L) const = 0;
	virtual void read(Base & o, lua_State * L, int index) const = 0;
	virtual void mtinit(lua_State * L) const = 0;
};

//generic field is an object, exposed as lightuserdata wrapped in a table
template<auto field>
struct Field : public FieldBase<typename Common::MemberPointer<decltype(field)>::Class> {
	using MP = Common::MemberPointer<decltype(field)>;
	using Base = typename MP::Class;
	using Value = typename MP::FieldType;

	virtual void push(Base & obj, lua_State * L) const override {
		LuaRW<Value>::push(L, obj.*field);
	}

	virtual void read(Base & obj, lua_State * L, int index) const override {
		obj.*field = LuaRW<Value>::read(L, index);
	}
	
	virtual void mtinit(lua_State * L) const override {
		if constexpr (std::is_class_v<Value>) {
			Info<Value>::mtinit(L);
		}
	}
};

template<>
struct Info<NeuralNet::Vector<double>> : public InfoBase<NeuralNet::Vector<double>> {
	using T = NeuralNet::Vector<double>;

	static constexpr char const * const mtname = "NeuralNet:Vector<double>";

	static std::map<std::string, FieldBase<T>*> fields;

	static constexpr luaL_Reg mtfields[] = {
		{"__index", ::__index<T>},
		{"__newindex", ::__newindex<T>},
		{"__len", ::__len<T>},
		{nullptr, nullptr},
	};

	static int __len(lua_State * L, T * o) {
		lua_pushinteger(L, o->size);
		return 1;
	}
};

std::map<std::string, FieldBase<NeuralNet::Vector<double>>*>
Info<NeuralNet::Vector<double>>::fields = {
	//{"size", new Field<&NeuralNet::Vector<double>::size>()},
	//{"storageSize", new Field<&NeuralNet::Vector<double>>()},
	//{"normL1", new Field<&NeuralNet::Vector<double>>()},
};


template<>
struct Info<NeuralNet::Matrix<double>> : public InfoBase<NeuralNet::Matrix<double>> {
	using T = NeuralNet::Matrix<double>;
	static constexpr char const * const mtname = "NeuralNet:Matrix<double>";
	static std::map<std::string, FieldBase<T>*> fields;
	static constexpr luaL_Reg mtfields[] = {
		{"__index", ::__index<T>},
		{"__newindex", ::__newindex<T>},
		{"__len", ::__len<T>},
		{nullptr, nullptr},
	};

	// Matrix # __len is its height
	// Matrix[i] will return the ThinVector of the row
	static int __len(lua_State * L, T * o) {
		lua_pushinteger(L, o->size.x);
		return 1;
	}
};


template<>
struct Info<NeuralNet::Layer<double>> : public InfoBase<NeuralNet::Layer<double>> {
	using T = NeuralNet::Layer<double>;
	static constexpr char const * const mtname = "NeuralNet::Layer<double>";
	static std::map<std::string, FieldBase<T>*> fields;
	static constexpr luaL_Reg mtfields[] = {
		{"__index", ::__index<T>},
		{"__newindex", ::__newindex<T>},
		{nullptr, nullptr},
	};
};
//static auto field_NeuralNet_Layer_x = Field<&NeuralNet::Layer<double>::x>();
//static auto field_NeuralNet_Layer_xErr = Field<&NeuralNet::Layer<double>::xErr>();
//static auto field_NeuralNet_Layer_w = Field<&NeuralNet::Layer<double>::w>();
//static auto field_NeuralNet_Layer_net = Field<&NeuralNet::Layer<double>::net>();
//static auto field_NeuralNet_Layer_netErr = Field<&NeuralNet::Layer<double>::netErr>();
//static auto field_NeuralNet_Layer_dw = Field<&NeuralNet::Layer<double>::dw>();
//static auto field_NeuralNet_Layer_activation = Field<&NeuralNet::Layer<double>::activation>();
//static auto field_NeuralNet_Layer_activationDeriv = Field<&NeuralNet::Layer<double>::activationDeriv>();
//static auto field_NeuralNet_Layer_getBias = Field<&NeuralNet::Layer<double>::getBias>();
std::map<std::string, FieldBase<NeuralNet::Layer<double>>*> Info<NeuralNet::Layer<double>>::fields = {
	// needs std::vector wrapper
	//{"x", &field_NeuralNet_Layer_x},
	//{"net", &field_NeuralNet_Layer_net},
	//{"w", &field_NeuralNet_Layer_w},
	//{"xErr", &field_NeuralNet_Layer_xErr},
	//{"netErr", &field_NeuralNet_Layer_netErr},
	//{"dw", &field_NeuralNet_Layer_dw},
	// needs func wrapper
	//{"activation", &field_NeuralNet_Layer_activation},
	//{"activationDeriv", &field_NeuralNet_Layer_activationDeriv},
	// needs method wrapper
	//{"getBias", &field_NeuralNet_Layer_getBias},
};


template<>
struct Info<NeuralNet::ANN<double>> 
: public InfoBase<NeuralNet::ANN<double>> 
{
	using T = NeuralNet::ANN<double>;
	
	static constexpr char const * const mtname = "NeuralNet::ANN<double>";
		
	static constexpr luaL_Reg mtfields[] = {
		{"__index", ::__index<T>},
		{"__newindex", ::__newindex<T>},
		{"__call", ::__call<T>},
		{nullptr, nullptr},
	};

	// call metatable = create new object
	// the member object access is lightuserdata i.e. no metatable ,so I'm wrapping it in a Lua table
	// so for consistency I'll do the same here ...
	static int __call(lua_State * L, T * o) {
		// stack: 1st arg should be the mt, since its call operator is the ann ctor
		int const nargs = lua_gettop(L);
		std::vector<int> layerSizes;
		for (int i = 2; i <= nargs; ++i) {
			layerSizes.push_back(lua_tointeger(L, i));
		}

		lua_newtable(L);
		luaL_setmetatable(L, Info<T>::mtname);
		new (L) T(layerSizes);
		lua_setfield(L, -2, "ptr");
		return 1;
	}
	
	static std::map<std::string, FieldBase<T>*> fields;
};

static auto field_NeuralNet_ANN_dt = Field<&NeuralNet::ANN<double>::dt>();
//static auto field_NeuralNet_ANN_layers = Field<&NeuralNet::ANN<double>::layers>();
static auto field_NeuralNet_ANN_useBatch = Field<&NeuralNet::ANN<double>::useBatch>();
static auto field_NeuralNet_ANN_batchCounter = Field<&NeuralNet::ANN<double>::batchCounter>();
static auto field_NeuralNet_ANN_totalBatchCounter = Field<&NeuralNet::ANN<double>::totalBatchCounter>();
//static auto field_NeuralNet_ANN_feedForward = Field<&NeuralNet::ANN<double>::feedForward>();
//static auto field_NeuralNet_ANN_calcError = Field<&NeuralNet::ANN<double>::calcError>();
//static auto field_NeuralNet_ANN_backPropagate = Field<&NeuralNet::ANN<double>::backPropagate>();
//static auto field_NeuralNet_ANN_updateBatch = Field<&NeuralNet::ANN<double>::updateBatch>();
//static auto field_NeuralNet_ANN_clearBatch = Field<&NeuralNet::ANN<double>::clearBatch>();
std::map<std::string, FieldBase<NeuralNet::ANN<double>>*> Info<NeuralNet::ANN<double>>::fields = {
	{"dt", &field_NeuralNet_ANN_dt},
	//{"layers", &field_NeuralNet_ANN_layers},
	{"useBatch", &field_NeuralNet_ANN_useBatch},
	{"batchCounter", &field_NeuralNet_ANN_batchCounter},
	{"totalBatchCounter", &field_NeuralNet_ANN_totalBatchCounter},
	//{"feedForward", &field_NeuralNet_ANN_feedForward},
	//{"calcError", &field_NeuralNet_ANN_calcError},
	//{"backPropagate", &field_NeuralNet_ANN_backPropagate},
	//{"updateBatch", &field_NeuralNet_ANN_updateBatch},
	//{"clearBatch", &field_NeuralNet_ANN_clearBatch},
};



int luaopen_NeuralNetLua(lua_State * L) {
	Info<NeuralNet::ANN<double>>::mtinit(L);

	luaL_getmetatable(L, Info<NeuralNet::ANN<double>>::mtname);
	return 1;
}
