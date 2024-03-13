// here's me trying to make c++ automated Lua binding
#include "NeuralNet/ANN.h"
#include <lua.hpp>
#include <type_traits>

/////////////////////////////////
// https://stackoverflow.com/a/59448568/2714073

namespace impl
{
/// Base declaration of our constexpr string_view concatenation helper
template <std::string_view const&, typename, std::string_view const&, typename>
struct concat;

/// Specialisation to yield indices for each char in both provided string_views,
/// allows us flatten them into a single char array
template <std::string_view const& S1,
          std::size_t... I1,
          std::string_view const& S2,
          std::size_t... I2>
struct concat<S1, std::index_sequence<I1...>, S2, std::index_sequence<I2...>>
{
  static constexpr const char value[]{S1[I1]..., S2[I2]..., 0};
};
} // namespace impl

/// Base definition for compile time joining of strings
template <std::string_view const&...> struct join;

/// When no strings are given, provide an empty literal
template <>
struct join<>
{
  static constexpr std::string_view value = "";
};

/// Base case for recursion where we reach a pair of strings, we concatenate
/// them to produce a new constexpr string
template <std::string_view const& S1, std::string_view const& S2>
struct join<S1, S2>
{
  static constexpr std::string_view value =
    impl::concat<S1,
                 std::make_index_sequence<S1.size()>,
                 S2,
                 std::make_index_sequence<S2.size()>>::value;
};

/// Main recursive definition for constexpr joining, pass the tail down to our
/// base case specialisation
template <std::string_view const& S, std::string_view const&... Rest>
struct join<S, Rest...>
{
  static constexpr std::string_view value =
    join<S, join<Rest...>::value>::value;
};

/// Join constexpr string_views to produce another constexpr string_view
template <std::string_view const&... Strs>
static constexpr auto join_v = join<Strs...>::value;

/////////////////////////////////


void * operator new(size_t size, lua_State * L) {
	return lua_newuserdata(L, size);
}

template<typename T>
struct Info;

// infos for prims.  doesn't have lua exposure, only mtname for mtname joining at compile time

template<>
struct Info<double> {
	static constexpr std::string_view mtname = "double";
};

// infos for structs:

template<typename T>
struct InfoStructBase {
	static void mtinit(lua_State * L) {
		if (luaL_newmetatable(L, Info<T>::mtname.data())) {
			luaL_setfuncs(L, Info<T>::mtfields, 0);
		}
		lua_pop(L, 1);
	
		for (auto & pair : Info<T>::getFields()) {
			pair.second->mtinit(L);
		}
	}

	// default behavior.  child template-specializations can override this.
	
	static int __index(lua_State * L, T * o) {
		char const * const k = lua_tostring(L, 2);
		auto const & fields = Info<T>::getFields();
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
		auto const & fields = Info<T>::getFields();
		auto iter = fields.find(k);
		if (iter == fields.end()) {
			luaL_error(L, "sorry, this table cannot accept new members");
		}
		iter->second->read(*o, L, 3);
		return 0;
	}
};

template<typename T>
struct Info {};

#define PTRFIELD "ptr"

template<typename T>
static T * lua_getptr(lua_State * L, int index) {
	luaL_checktype(L, index, LUA_TTABLE);
	lua_pushliteral(L, PTRFIELD);
	lua_rawget(L, index);
	luaL_checktype(L, -1, LUA_TUSERDATA);
	// verify metatable is Info<T>::mtname ... however lua_checkudata() does this but only for non-light userdata ...
	T * o = (T*)lua_touserdata(L, -1);
	lua_pop(L, 1);
	if (!o) luaL_error(L, "tried to index a null pointer");
	return o;
}

template<typename T>
static inline int __index(lua_State * L) {
	// use default which picks from Info::getFields()
	return Info<T>::__index(L, lua_getptr<T>(L, 1));
}

template<typename T>
static inline int __newindex(lua_State * L) {
	// use default which picks from Info::getFields()
	return Info<T>::__newindex(L, lua_getptr<T>(L, 1));
}

template<typename T>
static inline int __len(lua_State * L) {
	// if __len isn't defined in Info then this will give a compiler error
	return Info<T>::__len(L, lua_getptr<T>(L, 1));
}

template<typename T>
static inline int __call(lua_State * L) {
	// if __call isn't defined in Info then this will give a compiler error
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
		luaL_setmetatable(L, Info<T>::mtname.data());
		lua_pushliteral(L, PTRFIELD);
		lua_pushlightuserdata(L, &v);
		lua_rawset(L, -3);
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

template<typename Real>
struct Info<NeuralNet::Vector<Real>> 
: public InfoStructBase<NeuralNet::Vector<Real>> {
	using T = NeuralNet::Vector<Real>;

	static constexpr std::string_view strpre = "NeuralNet::Vector<";
	static constexpr std::string_view strsuf = ">";
	static constexpr std::string_view mtname = join_v<strpre, join_v<Info<Real>::mtname, strsuf>>;

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

	static auto & getFields() {
		static std::map<std::string, FieldBase<NeuralNet::Vector<Real>>*> fields = {
			//{"size", new Field<&NeuralNet::Vector<Real>::size>()},
			//{"storageSize", new Field<&NeuralNet::Vector<Real>>()},
			//{"normL1", new Field<&NeuralNet::Vector<Real>>()},
		};
		return fields;
	}
};


template<typename Real>
struct Info<NeuralNet::Matrix<Real>> 
: public InfoStructBase<NeuralNet::Matrix<Real>> {
	using T = NeuralNet::Matrix<Real>;
	
	static constexpr std::string_view strpre = "NeuralNet::Matrix<";
	static constexpr std::string_view strsuf = ">";
	static constexpr std::string_view mtname = join_v<strpre, join_v<Info<Real>::mtname, strsuf>>;
	
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
	
	static auto & getFields() {
		static std::map<std::string, FieldBase<T>*> fields;
		return fields;
	}
};


template<typename Real>
struct Info<NeuralNet::Layer<Real>> 
: public InfoStructBase<NeuralNet::Layer<Real>> {
	using T = NeuralNet::Layer<Real>;
	
	static constexpr std::string_view strpre = "NeuralNet::Layer<";
	static constexpr std::string_view strsuf = ">";
	static constexpr std::string_view mtname = join_v<strpre, join_v<Info<Real>::mtname, strsuf>>;

	static constexpr luaL_Reg mtfields[] = {
		{"__index", ::__index<T>},
		{"__newindex", ::__newindex<T>},
		{nullptr, nullptr},
	};
	
	static auto & getFields() {
		//static auto field_x = Field<&T::x>();
		//static auto field_xErr = Field<&T::xErr>();
		//static auto field_w = Field<&T::w>();
		//static auto field_net = Field<&T::net>();
		//static auto field_netErr = Field<&T::netErr>();
		//static auto field_dw = Field<&T::dw>();
		//static auto field_activation = Field<&T::activation>();
		//static auto field_activationDeriv = Field<&T::activationDeriv>();
		//static auto field_getBias = Field<&T::getBias>();
		static std::map<std::string, FieldBase<T>*> fields = {
			// needs std::vector wrapper
			//{"x", &field_x},
			//{"net", &field_net},
			//{"w", &field_w},
			//{"xErr", &field_xErr},
			//{"netErr", &field_netErr},
			//{"dw", &field_dw},
			// needs func wrapper
			//{"activation", &field_activation},
			//{"activationDeriv", &field_activationDeriv},
			// needs method wrapper
			//{"getBias", &field_getBias},
		};
		return fields;
	}
};

template<typename Real>
struct Info<NeuralNet::ANN<Real>> 
: public InfoStructBase<NeuralNet::ANN<Real>> {
	using Super = InfoStructBase<NeuralNet::ANN<Real>>;
	using T = NeuralNet::ANN<Real>;

	static constexpr std::string_view strpre = "NeuralNet::ANN<";
	static constexpr std::string_view strsuf = ">";
	static constexpr std::string_view mtname = join_v<strpre, join_v<Info<Real>::mtname, strsuf>>;
		
	static constexpr luaL_Reg mtfields[] = {
		{"__index", ::__index<T>},
		{"__newindex", ::__newindex<T>},
		{nullptr, nullptr},
	};

	// call metatable = create new object
	// the member object access is lightuserdata i.e. no metatable ,so I'm wrapping it in a Lua table
	// so for consistency I'll do the same with dense-userdata ...
	static int mt_new(lua_State * L) {
		// 1st arg is the metatable ... or its another ANN
		// stack: 1st arg should be the mt, since its call operator is the ann ctor
		int const nargs = lua_gettop(L);
		std::vector<int> layerSizes;
		for (int i = 2; i <= nargs; ++i) {
			layerSizes.push_back(lua_tointeger(L, i));
		}

		lua_newtable(L);
		luaL_setmetatable(L, Info<T>::mtname.data());
		lua_pushliteral(L, PTRFIELD);
		new(L) T(layerSizes);
		lua_rawset(L, -3);
		return 1;
	}
	
	static void mtinit(lua_State * L) {
		//init mt ...
		Super::mtinit(L);
	
		//then add call operator for ctor
		luaL_getmetatable(L, mtname.data());
		lua_pushcfunction(L, mt_new);
		lua_setfield(L, -2, "new");
		lua_pop(L, 1);
	}

	static auto & getFields() {
		static auto field_dt = Field<&T::dt>();
		//static auto field_layers = Field<&T::layers>();
		static auto field_useBatch = Field<&T::useBatch>();
		static auto field_batchCounter = Field<&T::batchCounter>();
		static auto field_totalBatchCounter = Field<&T::totalBatchCounter>();
		//static auto field_feedForward = Field<&T::feedForward>();
		//static auto field_calcError = Field<&T::calcError>();
		//static auto field_backPropagate = Field<&T::backPropagate>();
		//static auto field_updateBatch = Field<&T::updateBatch>();
		//static auto field_clearBatch = Field<&T::clearBatch>();
		static std::map<std::string, FieldBase<T>*> fields = {
			{"dt", &field_dt},
			//{"layers", &field_layers},
			{"useBatch", &field_useBatch},
			{"batchCounter", &field_batchCounter},
			{"totalBatchCounter", &field_totalBatchCounter},
			//{"feedForward", &field_feedForward},
			//{"calcError", &field_calcError},
			//{"backPropagate", &field_backPropagate},
			//{"updateBatch", &field_updateBatch},
			//{"clearBatch", &field_clearBatch},
		};
		return fields;
	}
};

extern "C" {
int luaopen_NeuralNetLua(lua_State * L) {
	Info<NeuralNet::ANN<double>>::mtinit(L);
	luaL_getmetatable(L, Info<NeuralNet::ANN<double>>::mtname.data());
	return 1;
}
}
