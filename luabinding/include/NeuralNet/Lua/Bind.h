#pragma once

#include "NeuralNet/Lua/join.h"
#include <lua.hpp>
#include <type_traits>


// "operator new cannot be declared inside a namespace"
// but I like this idea so ...
inline void * operator new(size_t size, lua_State * L) {
	return lua_newuserdata(L, size);
}

//TODO move to LuaCxx::Bind
namespace NeuralNet {
namespace Lua {

// here' all the generic lua-binding stuff

template<typename Type>
struct LuaBind;

// needs to be a macro and not a C++ typed expression for lua's pushliteral to work
#define LUACXX_BIND_PTRFIELD "ptr"

template<typename T>
static T * lua_getptr(lua_State * L, int index) {
	luaL_checktype(L, index, LUA_TTABLE);
	lua_pushliteral(L, LUACXX_BIND_PTRFIELD);
	lua_rawget(L, index);
	if (!lua_isuserdata(L, -1)) {	// both kinds of userdata plz
		luaL_checktype(L, -1, LUA_TUSERDATA);	// but i like their error so
	}
	// verify metatable is LuaBind<T>::mtname ... however lua_checkudata() does this but only for non-light userdata ...
	T * o = (T*)lua_touserdata(L, -1);
	lua_pop(L, 1);
	if (!o) luaL_error(L, "tried to index a null pointer");
	return o;
}

// I could just use the LuaBind static methods themselves, but meh, I cast the object here

template<typename T>
static inline int __index(lua_State * L) {
	return LuaBind<T>::__index(L, *lua_getptr<T>(L, 1));
}

template<typename T>
static inline int __newindex(lua_State * L) {
	return LuaBind<T>::__newindex(L, *lua_getptr<T>(L, 1));
}

template<typename T>
static inline int __len(lua_State * L) {
	return LuaBind<T>::__len(L, *lua_getptr<T>(L, 1));
}

template<typename T>
static inline int __call(lua_State * L) {
	return LuaBind<T>::__call(L, *lua_getptr<T>(L, 1));
}

template<typename T>
static inline int __tostring(lua_State * L) {
	return LuaBind<T>::__tostring(L, *lua_getptr<T>(L, 1));
}

// infos for prims.  doesn't have lua exposure, only mtname for mtname joining at compile time

template<>
struct LuaBind<double> {
	static constexpr std::string_view mtname = "double";
};

// default behavior.  child template-specializations can override this.

template<typename T>
static int default__index(lua_State * L) {
	auto & o = *lua_getptr<T>(L, 1);
	char const * const k = lua_tostring(L, 2);
	auto const & fields = LuaBind<T>::getFields();
	auto iter = fields.find(k);
	if (iter == fields.end()) {
		lua_pushnil(L);
		return 1;
	}
	iter->second->push(o, L);
	return 1;
}

template<typename T>
static int default__newindex(lua_State * L) {
	auto & o = *lua_getptr<T>(L, 1);
	char const * const k = lua_tostring(L, 2);
	assert(lua_gettop(L) == 3);	// t k v
	auto const & fields = LuaBind<T>::getFields();
	auto iter = fields.find(k);
	if (iter == fields.end()) {
#if 0	// option 1: no new fields
		luaL_error(L, "sorry, this table cannot accept new members");
#endif
#if 1	// option 2: write the Lua value into the Lua table
		lua_pushvalue(L, 2);
		lua_rawset(L, 1);
		return 0;
#endif
	}
	iter->second->read(o, L, 3);
	return 0;
}

template<typename T>
static int default__tostring(lua_State * L) {
	auto & o = *lua_getptr<T>(L, 1);
	lua_pushfstring(L, "%s: 0x%x", LuaBind<T>::mtname.data(), &o);
	return 1;
}

// base infos for all structs:
template<typename T> constexpr bool has__index_v = requires(T const & t) { t.__index; };
template<typename T> constexpr bool has__newindex_v = requires(T const & t) { t.__newindex; };
template<typename T> constexpr bool has__len_v = requires(T const & t) { t.__len; };
template<typename T> constexpr bool has__call_v = requires(T const & t) { t.__call; };
template<typename T> constexpr bool has__tostring_v = requires(T const & t) { t.__tostring; };
template<typename T> constexpr bool has_mt_ctor_v = requires(T const & t) { t.mt_ctor; };

template<typename T>
struct LuaBindStructBase {

	// add the class constructor as the call operator of the metatable
	static void initMtCtor(lua_State * L) {
		static constexpr std::string_view suffix = " metatable";
		static constexpr std::string_view mtname = join_v<LuaBind<T>::mtname, suffix>;
		if (!luaL_newmetatable(L, mtname.data())) return;

		lua_pushcfunction(L, LuaBind<T>::mt_ctor);
		lua_setfield(L, -2, "__call");
	}

	/*
	initialize the metatable associated with this type
	*/
	static void mtinit(lua_State * L) {
		auto const & mtname = LuaBind<T>::mtname;

		for (auto & pair : LuaBind<T>::getFields()) {
			pair.second->mtinit(L);
		}

		if (luaL_newmetatable(L, mtname.data())) {
			// not supported in luajit ...
			lua_pushstring(L, mtname.data());
			lua_setfield(L, -2, "__name");

//std::cout << "building " << mtname << " metatable" << std::endl;

			/*
			for __index and __newindex I'm providing default behavior.
			i used to provide it via static parent class method, but that emant overriding children ha to always 'using' to specify their own impl ver the aprent
			so nw i'm just moving the defult outside the class
			but the downside is - with the if constexpr check here - that there will always be an __index but maybe that wasla ways the case ...
			*/

			if constexpr (has__tostring_v<LuaBind<T>>) {
				lua_pushcfunction(L, ::NeuralNet::Lua::__tostring<T>);
				lua_setfield(L, -2, "__tostring");
//std::cout << "assigning __tostring" << std::endl;
			} else {
				lua_pushcfunction(L, ::NeuralNet::Lua::default__tostring<T>);
				lua_setfield(L, -2, "__tostring");
//std::cout << "using default __tostring" << std::endl;
			}

			if constexpr (has__index_v<LuaBind<T>>) {
				lua_pushcfunction(L, ::NeuralNet::Lua::__index<T>);
				lua_setfield(L, -2, "__index");
//std::cout << "assigning __index" << std::endl;
			} else {
				lua_pushcfunction(L, ::NeuralNet::Lua::default__index<T>);
				lua_setfield(L, -2, "__index");
//std::cout << "using default __index" << std::endl;
			}

			if constexpr (has__newindex_v<LuaBind<T>>) {
				lua_pushcfunction(L, ::NeuralNet::Lua::__newindex<T>);
				lua_setfield(L, -2, "__newindex");
//std::cout << "assigning __newindex" << std::endl;
			} else {
				lua_pushcfunction(L, ::NeuralNet::Lua::default__newindex<T>);
				lua_setfield(L, -2, "__newindex");
//std::cout << "using default __newindex" << std::endl;
			}

			// the rest of these are optional to provide

			if constexpr (has__len_v<LuaBind<T>>) {
				lua_pushcfunction(L, ::NeuralNet::Lua::__len<T>);
				lua_setfield(L, -2, "__len");
//std::cout << "assigning __len" << std::endl;
			}

			if constexpr (has__call_v<LuaBind<T>>) {
				lua_pushcfunction(L, ::NeuralNet::Lua::__call<T>);
				lua_setfield(L, -2, "__call");
//std::cout << "assigning __call" << std::endl;
			}

			// ok now let's give the metatable a metatable, so if someone calls it, it'll call the ctor
			if constexpr (has_mt_ctor_v<LuaBind<T>>) {
				initMtCtor(L);
				lua_setmetatable(L, -2);
//std::cout << "assigning metatable __call ctor" << std::endl;
			}
		}
		lua_pop(L, 1);
	}
};


// general case just wraps the memory
template<typename T>
struct LuaRW {
	static void push(lua_State * L, T & v) {
		// hmm, we want a metatable for what we return
		// but lightuserdata has no metatable
		// so it'll have to be a new lua table that points back to this
		lua_newtable(L);
#if 1	// if the metatable isn't there then it won't be set
		luaL_setmetatable(L, LuaBind<T>::mtname.data());
#endif
#if 0 	//isn't this supposed to do the same as luaL_setmetatable ?
		luaL_getmetatable(L, LuaBind<T>::mtname.data());
std::cout << "metatable " << LuaBind<T>::mtname << " type " << lua_type(L, -1) << std::endl;
		lua_setmetatable(L, -2);
#endif
#if 0 // this say ssomething is there, but it always returns nil
		lua_getmetatable(L, -1);
std::cout << "metatable " << LuaBind<T>::mtname << " type " << lua_type(L, -1) << std::endl;
		lua_pop(L, 1);
#endif
		lua_pushliteral(L, LUACXX_BIND_PTRFIELD);
		lua_pushlightuserdata(L, &v);
		lua_rawset(L, -3);
	}

	static T read(lua_State * L, int index) {
		luaL_error(L, "this field cannot be overwritten");
		throw std::runtime_error("this field cannot be overwritten");
		//return {};	// hmm this needs the default ctor to exist, but I'm throwing,
					// so it doesn't really need to exist ...
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

// should this go in Common/MemberPointer.h?
// what's the better way to do this?
template<typename T>
struct MemberBaseClass {
	static decltype(auto) value() {
		if constexpr(std::is_member_object_pointer_v<T>) {
			using type = typename Common::MemberPointer<T>::Class;
			return (type*)nullptr;
		} else if constexpr (std::is_member_function_pointer_v<T>) {
			using type = typename Common::MemberMethodPointer<T>::Class;
			return (type*)nullptr;
		} else {
			return nullptr;
		}
	}
	using type = typename std::remove_pointer_t<decltype(value())>;\
};



template<auto field>
int memberMethodWrapper(lua_State * L) {
	using MMP = Common::MemberMethodPointer<decltype(field)>;
	using Args = typename MMP::Args;
	using Return = typename MMP::Return;
	auto & o = *lua_getptr<typename MMP::Class>(L, 1);
	// TODO here, and in Matrix returning ThinVector
	// and maybe overall ...
	// ... if we are writing by value then push a dense userdata
	// ... if we are writing by pointer or ref then push a light userdata
	// but mind you, how to determine from LuaRW, unless we only use pass-by-value for values and pass-by-ref for refs ?
	// or maybe another template arg in LuaRW?

	if constexpr (std::is_same_v<Return, void>) {
		[&]<auto...j>(std::index_sequence<j...>) -> decltype(auto) {
			(o.*field)(
				LuaRW<std::tuple_element_t<j, Args>>::read(L, j+1)...
			);
		}(std::make_index_sequence<std::tuple_size_v<Args>>{});
		return 0;
	} else {
		LuaRW<Return>::push(L,
			[&]<auto...j>(std::index_sequence<j...>) -> decltype(auto) {
				return (o.*field)(
					LuaRW<std::tuple_element_t<j, Args>>::read(L, j+1)...
				);
			}(std::make_index_sequence<std::tuple_size_v<Args>>{})
		);
		return 1;
	}
}

//generic field is an object, exposed as lightuserdata wrapped in a table
template<auto field>
struct Field
: public FieldBase<typename MemberBaseClass<decltype(field)>::type>
{
	using T = decltype(field);
	using Base = typename MemberBaseClass<T>::type;

	virtual void push(Base & obj, lua_State * L) const override {
		if constexpr(std::is_member_object_pointer_v<T>) {
			using Value = typename Common::MemberPointer<T>::FieldType;
			LuaRW<Value>::push(L, obj.*field);
		} else if (std::is_member_function_pointer_v<T>) {
			//push a c function that calls the member method (and transforms all the arguments)
			lua_pushcfunction(L, memberMethodWrapper<field>);
		}
	}

	virtual void read(Base & obj, lua_State * L, int index) const override {
		if constexpr(std::is_member_object_pointer_v<T>) {
			using Value = typename Common::MemberPointer<T>::FieldType;
			obj.*field = LuaRW<Value>::read(L, index);
		} else if (std::is_member_function_pointer_v<T>) {
			luaL_error(L, "this field is read only");
		}
	}

	virtual void mtinit(lua_State * L) const override {
		if constexpr(std::is_member_object_pointer_v<T>) {
			using Value = typename Common::MemberPointer<T>::FieldType;
			if constexpr (std::is_class_v<Value>) {
				LuaBind<Value>::mtinit(L);
			}
		} else if (std::is_member_function_pointer_v<T>) {
			using Return = typename Common::MemberMethodPointer<T>::Return;
			if constexpr (std::is_class_v<Return>) {
				LuaBind<Return>::mtinit(L);
			}
			// need to add arg types too or nah?  nah... returns are returned, so they could be a first creation of that type. not args.
		}
	}
};

#if 0
template<auto field>
requires (std::is_member_function_pointer_v<decltype(field)>)
struct Field : public FieldBase<
	typename Common::MemberMethodPointer<decltype(field)>::Class
> {
	using MP = Common::MemberMethodPointer<decltype(field);
	using Base = typename MP::Class;
	using Return = typename MP::Return;

	virtual void push(Base & obj, lua_State * L) const override {
		//LuaRW<Return>::push(L, obj.*field);
		lua_pushcfunction(L, field);
	}

	virtual void read(Base & obj, lua_State * L, int index) const override {
		//obj.*field = LuaRW<Return>::read(L, index);
		luaL_error(L, "field is read-only");
	}

	virtual void mtinit(lua_State * L) const override {
		if constexpr (std::is_class_v<Return>) {
			LuaBind<Return>::mtinit(L);
		}
	}
};
#endif

// generalized __len, __index, __newindex access

// child needs to provide IndexAccessRead, IndexAccessWrite, IndexLen
template<typename CRTPChild, typename Type, typename Elem>
struct IndexAccessReadWrite {
	static int __index(lua_State * L, Type & o) {
		if (lua_type(L, 2) != LUA_TNUMBER) {
			lua_pushnil(L);
			return 1;
		}
		int i = lua_tointeger(L, 2);
		--i;
		// using 1-based indexing. sue me.
		if (i < 0 || i >= CRTPChild::IndexLen(o)) {
			lua_pushnil(L);
			return 1;
		}
		CRTPChild::IndexAccessRead(L, o, i);
		return 1;
	}

	static int __newindex(lua_State * L, Type & o) {
		if (lua_type(L, 2) != LUA_TNUMBER) {
			luaL_error(L, "can only write to index elements");
		}
		int i = lua_tointeger(L, 2);
		--i;
		// using 1-based indexing. sue me.
		if (i < 0 || i >= CRTPChild::IndexLen(o)) {
			luaL_error(L, "index %d is out of bounds", i+1);
		}
		CRTPChild::IndexAccessWrite(L, o, i);
		return 1;
	}

	static int __len(lua_State * L, Type & o) {
		lua_pushinteger(L, CRTPChild::IndexLen(o));
		return 1;
	}
};

// CRTPChild needs to provide IndexAt, IndexLen
template<typename CRTPChild, typename Type, typename Elem>
struct IndexAccess
: public IndexAccessReadWrite<
	IndexAccess<CRTPChild, Type, Elem>,	// pass the IndexAccess as the new CRTPChild so it can see the IndexAccessRead and IndexAccessWrite here
	Type,
	Elem
>
{
	static void IndexAccessRead(lua_State * L, Type & o, int i) {
		LuaRW<Elem>::push(L, CRTPChild::IndexAt(L, o, i));
	}

	static void IndexAccessWrite(lua_State * L, Type & o, int i) {
		// will error if you try to write a non-prim
		CRTPChild::IndexAt(L, o, i) = LuaRW<Elem>::read(L, 3);
	}

	//use CRTPChild's IndexLen
	static int IndexLen(Type const & o) {
		return CRTPChild::IndexLen(o);
	}
};

// infos for stl

template<typename Elem>
struct LuaBind<std::vector<Elem>>
:	public LuaBindStructBase<std::vector<Elem>>,
	public IndexAccess<
		LuaBind<std::vector<Elem>>,
		std::vector<Elem>,
		Elem
	>
{
	using Super = LuaBindStructBase<std::vector<Elem>>;
	using Type = std::vector<Elem>;

	static constexpr std::string_view strpre = "std::vector<";
	static constexpr std::string_view strsuf = ">";
	static constexpr std::string_view mtname = join_v<strpre, join_v<LuaBind<Elem>::mtname, strsuf>>;

	// vector needs Elem's metatable initialized
	static void mtinit(lua_State * L) {
		Super::mtinit(L);

		//init all subtypes
		//this test is same as in Field::mtinit
		if constexpr (std::is_class_v<Elem>) {
			LuaBind<Elem>::mtinit(L);
		}
	}

	static Elem & IndexAt(lua_State * L, Type & o, int i) {
		return o[i];
	}

	static int IndexLen(Type const & o) {
		return o.size();
	}

	static auto & getFields() {
		static std::map<std::string, FieldBase<Type>*> fields = {
		};
		return fields;
	}
};

}
}
