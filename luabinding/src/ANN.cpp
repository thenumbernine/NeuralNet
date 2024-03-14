// here's me trying to make c++ automated Lua binding
#include "NeuralNet/ANN.h"
#include "NeuralNet/Lua/Bind.h"


// info for ANN structs:

template<typename Real>
struct NeuralNet::Lua::Bind<NeuralNet::Vector<Real>>
:	public BindStructBase<NeuralNet::Vector<Real>>,
	public IndexAccess<
		NeuralNet::Lua::Bind<NeuralNet::Vector<Real>>,
		NeuralNet::Vector<Real>,
		Real
	>
{
	using Type = NeuralNet::Vector<Real>;

	static constexpr std::string_view strpre = "NeuralNet::Vector<";
	static constexpr std::string_view strsuf = ">";
	static constexpr std::string_view mtname = join_v<strpre, join_v<NeuralNet::Lua::Bind<Real>::mtname, strsuf>>;

	static Real & IndexAt(lua_State * L, Type & o, int i) {
		return o[i];
	}

	static int IndexLen(Type const & o) {
		return o.size;
	}

	static auto & getFields() {
		static auto field_normL1 = Field<&Type::normL1>();
		static std::map<std::string, FieldBase<Type>*> fields = {
			//{"size", new Field<&NeuralNet::Vector<Real>::size>()},
			//{"storageSize", new Field<&NeuralNet::Vector<Real>>()},
			{"normL1", &field_normL1},
		};
		return fields;
	}
};

template<typename Real>
struct NeuralNet::Lua::Bind<NeuralNet::ThinVector<Real>>
:	public BindStructBase<NeuralNet::ThinVector<Real>>,
	public IndexAccess<
		NeuralNet::Lua::Bind<NeuralNet::ThinVector<Real>>,
		NeuralNet::ThinVector<Real>,
		Real
	>
{
	using Super = BindStructBase<NeuralNet::ThinVector<Real>>;
	using Type = NeuralNet::ThinVector<Real>;

	static constexpr std::string_view strpre = "NeuralNet::ThinVector<";
	static constexpr std::string_view strsuf = ">";
	static constexpr std::string_view mtname = join_v<strpre, join_v<NeuralNet::Lua::Bind<Real>::mtname, strsuf>>;

	static Real & IndexAt(lua_State * L, Type & o, int i) {
		return o[i];
	}

	static int IndexLen(Type const & o) {
		return o.size;
	}

	static auto & getFields() {
		static auto field_normL1 = Field<&Type::normL1>();
		static std::map<std::string, FieldBase<Type>*> fields = {
			//{"size", new Field<&NeuralNet::Vector<Real>::size>()},
			//{"storageSize", new Field<&NeuralNet::Vector<Real>>()},
			{"normL1", &field_normL1},
		};
		return fields;
	}
};


template<typename Real>
struct NeuralNet::Lua::Bind<NeuralNet::Matrix<Real>>
:	public BindStructBase<NeuralNet::Matrix<Real>>,
	public IndexAccessReadWrite<
		NeuralNet::Lua::Bind<NeuralNet::Matrix<Real>>,
		NeuralNet::Matrix<Real>,
		Real
	>
{
	using Super = BindStructBase<NeuralNet::Matrix<Real>>;
	using Type = NeuralNet::Matrix<Real>;

	static constexpr std::string_view strpre = "NeuralNet::Matrix<";
	static constexpr std::string_view strsuf = ">";
	static constexpr std::string_view mtname = join_v<strpre, join_v<NeuralNet::Lua::Bind<Real>::mtname, strsuf>>;

	// vector needs Elem's metatable initialized
	static void mtinit(lua_State * L) {
		Super::mtinit(L);

		//init all subtypes
		NeuralNet::Lua::Bind<NeuralNet::ThinVector<Real>>::mtinit(L);
	}

	// create a full-userdata of the ThinVector so that it sticks around when Lua tries to access it
	static void IndexAccessRead(lua_State * L, Type & o, int i) {
		lua_newtable(L);
		luaL_setmetatable(L, NeuralNet::Lua::Bind<NeuralNet::ThinVector<Real>>::mtname.data());
		lua_pushliteral(L, LUACXX_BIND_PTRFIELD);
		new(L) NeuralNet::ThinVector(o[i]);
		lua_rawset(L, -3);
	}

	static void IndexAccessWrite(lua_State * L, Type & o, int i) {
		lua_newtable(L);
		luaL_setmetatable(L, NeuralNet::Lua::Bind<NeuralNet::ThinVector<Real>>::mtname.data());
		lua_pushliteral(L, LUACXX_BIND_PTRFIELD);
		new(L) NeuralNet::ThinVector(o[i]);
		lua_rawset(L, -3);
	}

	static int IndexLen(Type & o) {
		return o.size.x;
	}

	static auto & getFields() {
		static std::map<std::string, FieldBase<Type>*> fields;
		return fields;
	}
};


template<typename Real>
struct NeuralNet::Lua::Bind<NeuralNet::Layer<Real>>
: public BindStructBase<NeuralNet::Layer<Real>> {
	using Type = NeuralNet::Layer<Real>;

	static constexpr std::string_view strpre = "NeuralNet::Layer<";
	static constexpr std::string_view strsuf = ">";
	static constexpr std::string_view mtname = join_v<strpre, join_v<NeuralNet::Lua::Bind<Real>::mtname, strsuf>>;

	static auto & getFields() {
		static auto field_x = Field<&Type::x>();
		static auto field_xErr = Field<&Type::xErr>();
		static auto field_w = Field<&Type::w>();
		static auto field_net = Field<&Type::net>();
		static auto field_netErr = Field<&Type::netErr>();
		static auto field_dw = Field<&Type::dw>();
		//static auto field_activation = Field<&Type::activation>();
		//static auto field_activationDeriv = Field<&Type::activationDeriv>();
		static auto field_getBias = Field<&Type::getBias>();
		static std::map<std::string, FieldBase<Type>*> fields = {
			{"x", &field_x},
			{"net", &field_net},
			{"w", &field_w},
			{"xErr", &field_xErr},
			{"netErr", &field_netErr},
			{"dw", &field_dw},
			// needs func wrapper
			//{"activation", &field_activation},
			//{"activationDeriv", &field_activationDeriv},
			// needs method wrapper
			{"getBias", &field_getBias},
		};
		return fields;
	}
};

template<typename Real>
struct NeuralNet::Lua::Bind<NeuralNet::ANN<Real>>
: public BindStructBase<NeuralNet::ANN<Real>> {
	using Super = BindStructBase<NeuralNet::ANN<Real>>;
	using Type = NeuralNet::ANN<Real>;

	static constexpr std::string_view strpre = "NeuralNet::ANN<";
	static constexpr std::string_view strsuf = ">";
	static constexpr std::string_view mtname = join_v<strpre, join_v<NeuralNet::Lua::Bind<Real>::mtname, strsuf>>;

	// call metatable = create new object
	// the member object access is lightuserdata i.e. no metatable ,so I'm wrapping it in a Lua table
	// so for consistency I'll do the same with dense-userdata ...
	static int mt_ctor(lua_State * L) {
		// TODO would be nice to abstract this into ctor arg interpretation
		// then I could move mt_ctor to the BindStructBase parent
		// 1st arg is the metatable ... or its another ANN
		// stack: 1st arg should be the mt, since its call operator is the ann ctor

		// TODO ANN has an initializer_list ctor ... woudl be nice to jsut fwd args like I'm doing for the call wrapper ...
		int const nargs = lua_gettop(L);
		std::vector<int> layerSizes;
		for (int i = 2; i <= nargs; ++i) {
			layerSizes.push_back(lua_tointeger(L, i));
		}

		lua_newtable(L);
		luaL_setmetatable(L, NeuralNet::Lua::Bind<Type>::mtname.data());
		lua_pushliteral(L, LUACXX_BIND_PTRFIELD);
		new(L) Type(layerSizes);
		lua_rawset(L, -3);
		return 1;
	}

	static auto & getFields() {
		// TODO autogen from fields[] tuple
		// maybe even use that tuple instead of this map ...
		// ... but ... runtime O(log(n)) map access vs compile-time O(n) tuple iteration ... map still wins
		// but what about TODO compile-time O(log(n)) recursive tree access of fields
		static auto field_layers = Field<&Type::layers>();
		static auto field_output = Field<&Type::output>();
		static auto field_outputError = Field<&Type::outputError>();
		static auto field_desired = Field<&Type::desired>();
		static auto field_dt = Field<&Type::dt>();
		static auto field_useBatch = Field<&Type::useBatch>();
		static auto field_batchCounter = Field<&Type::batchCounter>();
		static auto field_totalBatchCounter = Field<&Type::totalBatchCounter>();
		// TODO member functions that return refs
		//static auto field_input = Field<&Type::input>();
		//static auto field_inputError = Field<&Type::inputError>();
		static auto field_feedForward = Field<&Type::feedForward>();
		static auto field_calcError = Field<&Type::calcError>();
		static auto field_backPropagate = Field<
			static_cast<void (Type::*)()>(&Type::backPropagate)
		>();
		static auto field_backPropagate_dt = Field<
			static_cast<void (Type::*)(Real)>(&Type::backPropagate)
		>();
		static auto field_updateBatch = Field<&Type::updateBatch>();
		static auto field_clearBatch = Field<&Type::clearBatch>();
		static std::map<std::string, FieldBase<Type>*> fields = {
			{"layers", &field_layers},
			{"output", &field_output},
			{"outputError", &field_outputError},
			{"desired", &field_desired},
			{"dt", &field_dt},
			{"useBatch", &field_useBatch},
			{"batchCounter", &field_batchCounter},
			{"totalBatchCounter", &field_totalBatchCounter},

			//{"input", &field_input},
			//{"inputError", &field_inputError},
			{"feedForward", &field_feedForward},
			{"calcError", &field_calcError},

			{"backPropagate", &field_backPropagate},

			// TODO would be nice for the binding to also handle overloads ... maybe some day
			{"backPropagate_dt", &field_backPropagate_dt},

			{"updateBatch", &field_updateBatch},
			{"clearBatch", &field_clearBatch},
		};
		return fields;
	}
};



#include <stdfloat>
#if defined(__STDCPP_FLOAT16_T__)
template<> struct NeuralNet::Lua::Bind<std::float16_t> { static constexpr std::string_view mtname = "std::float16_t"; };
#endif
#if defined(__STDCPP_FLOAT32_T__)
template<> struct NeuralNet::Lua::Bind<std::float32_t> { static constexpr std::string_view mtname = "std::float32_t"; };
#endif
#if defined(__STDCPP_FLOAT64_T__)
template<> struct NeuralNet::Lua::Bind<std::float64_t> { static constexpr std::string_view mtname = "std::float64_t"; };
#endif
#if defined(__STDCPP_FLOAT128_T__)
template<> struct NeuralNet::Lua::Bind<std::float128_t> { static constexpr std::string_view mtname = "std::float128_t"; };
#endif
#if defined(__STDCPP_BFLOAT16_T__)
template<> struct NeuralNet::Lua::Bind<std::bfloat16_t> { static constexpr std::string_view mtname = "std::bfloat16_t"; };
#endif

extern "C" {
int luaopen_NeuralNetLua(lua_State * L) {
	// instanciate as many template types as you want here
	// don't be shy, add some int types while you're at it
	using types = std::tuple<
		NeuralNet::ANN<float>,
		NeuralNet::ANN<double>,
		NeuralNet::ANN<long double>
#if defined(__STDCPP_FLOAT16_T__)
		,NeuralNet::ANN<std::float16_t>
#endif
#if defined(__STDCPP_FLOAT32_T__)
		,NeuralNet::ANN<std::float32_t>
#endif
#if defined(__STDCPP_FLOAT64_T__)
		,NeuralNet::ANN<std::float64_t>
#endif
#if defined(__STDCPP_FLOAT128_T__)
		,NeuralNet::ANN<std::float128_t>
#endif
#if defined(__STDCPP_BFLOAT16_T__)
		,NeuralNet::ANN<std::bfloat16_t>
#endif
	>;

	// if I inline the lambda def then I get "error: use 'template' keyword to treat 'operator ()' as a dependent template name"
	// so I guess it has to sit here outside the loop
	auto buildType = [&]<typename T>() constexpr {
		using Bind = NeuralNet::Lua::Bind<T>;
		Bind::mtinit(L);
		auto name = Bind::mtname;
		luaL_getmetatable(L, name.data());
		lua_setfield(L, -2, name.data());
	};

	lua_newtable(L);
	[&]<auto...j>(std::index_sequence<j...>) constexpr {
		(buildType.operator()<std::tuple_element_t<j, types>>(), ...);
	}(std::make_index_sequence<std::tuple_size_v<types>>{});
	return 1;
}
}
