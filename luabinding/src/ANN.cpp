// here's me trying to make c++ automated Lua binding
#include "NeuralNet/ANN.h"
#include "NeuralNet/Lua/Bind.h"


// info for ANN structs:

template<typename Real>
struct NeuralNet::Lua::LuaBind<NeuralNet::Vector<Real>>
:	public LuaBindStructBase<NeuralNet::Vector<Real>>,
	public IndexAccess<
		NeuralNet::Lua::LuaBind<NeuralNet::Vector<Real>>,
		NeuralNet::Vector<Real>,
		Real
	>
{
	using Type = NeuralNet::Vector<Real>;

	static constexpr std::string_view strpre = "NeuralNet::Vector<";
	static constexpr std::string_view strsuf = ">";
	static constexpr std::string_view mtname = join_v<strpre, join_v<NeuralNet::Lua::LuaBind<Real>::mtname, strsuf>>;

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
struct NeuralNet::Lua::LuaBind<NeuralNet::ThinVector<Real>>
:	public LuaBindStructBase<NeuralNet::ThinVector<Real>>,
	public IndexAccess<
		NeuralNet::Lua::LuaBind<NeuralNet::ThinVector<Real>>,
		NeuralNet::ThinVector<Real>,
		Real
	>
{
	using Super = LuaBindStructBase<NeuralNet::ThinVector<Real>>;
	using Type = NeuralNet::ThinVector<Real>;

	static constexpr std::string_view strpre = "NeuralNet::ThinVector<";
	static constexpr std::string_view strsuf = ">";
	static constexpr std::string_view mtname = join_v<strpre, join_v<NeuralNet::Lua::LuaBind<Real>::mtname, strsuf>>;

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
struct NeuralNet::Lua::LuaBind<NeuralNet::Matrix<Real>>
:	public LuaBindStructBase<NeuralNet::Matrix<Real>>,
	public IndexAccessReadWrite<
		NeuralNet::Lua::LuaBind<NeuralNet::Matrix<Real>>,
		NeuralNet::Matrix<Real>,
		Real
	>
{
	using Super = LuaBindStructBase<NeuralNet::Matrix<Real>>;
	using Type = NeuralNet::Matrix<Real>;

	static constexpr std::string_view strpre = "NeuralNet::Matrix<";
	static constexpr std::string_view strsuf = ">";
	static constexpr std::string_view mtname = join_v<strpre, join_v<NeuralNet::Lua::LuaBind<Real>::mtname, strsuf>>;

	// vector needs Elem's metatable initialized
	static void mtinit(lua_State * L) {
		Super::mtinit(L);

		//init all subtypes
		NeuralNet::Lua::LuaBind<NeuralNet::ThinVector<Real>>::mtinit(L);
	}

	// create a full-userdata of the ThinVector so that it sticks around when Lua tries to access it
	static void IndexAccessRead(lua_State * L, Type & o, int i) {
		lua_newtable(L);
		luaL_setmetatable(L, NeuralNet::Lua::LuaBind<NeuralNet::ThinVector<Real>>::mtname.data());
		lua_pushliteral(L, LUACXX_BIND_PTRFIELD);
		new(L) NeuralNet::ThinVector(o[i]);
		lua_rawset(L, -3);
	}

	static void IndexAccessWrite(lua_State * L, Type & o, int i) {
		lua_newtable(L);
		luaL_setmetatable(L, NeuralNet::Lua::LuaBind<NeuralNet::ThinVector<Real>>::mtname.data());
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
struct NeuralNet::Lua::LuaBind<NeuralNet::Layer<Real>>
: public LuaBindStructBase<NeuralNet::Layer<Real>> {
	using Type = NeuralNet::Layer<Real>;

	static constexpr std::string_view strpre = "NeuralNet::Layer<";
	static constexpr std::string_view strsuf = ">";
	static constexpr std::string_view mtname = join_v<strpre, join_v<NeuralNet::Lua::LuaBind<Real>::mtname, strsuf>>;

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
struct NeuralNet::Lua::LuaBind<NeuralNet::ANN<Real>>
: public LuaBindStructBase<NeuralNet::ANN<Real>> {
	using Super = LuaBindStructBase<NeuralNet::ANN<Real>>;
	using Type = NeuralNet::ANN<Real>;

	static constexpr std::string_view strpre = "NeuralNet::ANN<";
	static constexpr std::string_view strsuf = ">";
	static constexpr std::string_view mtname = join_v<strpre, join_v<NeuralNet::Lua::LuaBind<Real>::mtname, strsuf>>;

	// call metatable = create new object
	// the member object access is lightuserdata i.e. no metatable ,so I'm wrapping it in a Lua table
	// so for consistency I'll do the same with dense-userdata ...
	static int mt_ctor(lua_State * L) {
		// TODO would be nice to abstract this into ctor arg interpretation
		// then I could move mt_ctor to the LuaBindStructBase parent
		// 1st arg is the metatable ... or its another ANN
		// stack: 1st arg should be the mt, since its call operator is the ann ctor
		
		// TODO ANN has an initializer_list ctor ... woudl be nice to jsut fwd args like I'm doing for the call wrapper ...
		int const nargs = lua_gettop(L);
		std::vector<int> layerSizes;
		for (int i = 2; i <= nargs; ++i) {
			layerSizes.push_back(lua_tointeger(L, i));
		}

		lua_newtable(L);
		luaL_setmetatable(L, NeuralNet::Lua::LuaBind<Type>::mtname.data());
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
		static auto field_dt = Field<&Type::dt>();
		static auto field_layers = Field<&Type::layers>();
		static auto field_useBatch = Field<&Type::useBatch>();
		static auto field_batchCounter = Field<&Type::batchCounter>();
		static auto field_totalBatchCounter = Field<&Type::totalBatchCounter>();
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
			{"dt", &field_dt},
			{"layers", &field_layers},
			{"useBatch", &field_useBatch},
			{"batchCounter", &field_batchCounter},
			{"totalBatchCounter", &field_totalBatchCounter},
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

extern "C" {
int luaopen_NeuralNetLua(lua_State * L) {
	
	// instanciate as many template types as you want here
	NeuralNet::Lua::LuaBind<NeuralNet::ANN<double>>::mtinit(L);
	
	luaL_getmetatable(L, NeuralNet::Lua::LuaBind<NeuralNet::ANN<double>>::mtname.data());
	return 1;
}
}
