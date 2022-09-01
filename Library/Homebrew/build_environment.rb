# typed: true
# frozen_string_literal: true

# Settings for the build environment.
#
# @api private
class BuildEnvironment
  extend T::Sig

  sig { params(settings: Symbol).void }
  def initialize(*settings)
    @settings = Set.new(settings)
  end

  sig { params(args: T::Enumerable[Symbol]).returns(T.self_type) }
  def merge(*args)
    @settings.merge(*args)
    self
  end

  sig { params(o: Symbol).returns(T.self_type) }
  def <<(o)
    @settings << o
    self
  end

  sig { returns(T::Boolean) }
  def std?
    @settings.include? :std
  end

  # DSL for specifying build environment settings.
  module DSL
    extend T::Sig

    # Initialise @env for each class which may use this DSL (e.g. each formula subclass).
    # `env` may never be called, and it needs to be initialised before the class is frozen.
    def inherited(child)
      super
      child.instance_eval do
        @env = BuildEnvironment.new
      end
    end

    sig { params(settings: Symbol).returns(BuildEnvironment) }
    def env(*settings)
      @env.merge(settings)
    end
  end

  KEYS = %w[
    CC CXX LD OBJC OBJCXX
    HOMEBREW_CC HOMEBREW_CXX
    CFLAGS CXXFLAGS CPPFLAGS LDFLAGS SDKROOT MAKEFLAGS
    CMAKE_PREFIX_PATH CMAKE_INCLUDE_PATH CMAKE_LIBRARY_PATH CMAKE_FRAMEWORK_PATH
    MACOSX_DEPLOYMENT_TARGET PKG_CONFIG_PATH PKG_CONFIG_LIBDIR
    HOMEBREW_DEBUG HOMEBREW_MAKE_JOBS HOMEBREW_VERBOSE
    HOMEBREW_SVN HOMEBREW_GIT
    HOMEBREW_SDKROOT
    MAKE GIT CPP
    ACLOCAL_PATH PATH CPATH
    LD_LIBRARY_PATH LD_RUN_PATH LD_PRELOAD LIBRARY_PATH
  ].freeze
  private_constant :KEYS

  sig { params(env: T::Hash[String, T.nilable(T.any(String, Pathname))]).returns(T::Array[String]) }
  def self.keys(env)
    KEYS & env.keys
  end

  sig { params(env: T::Hash[String, T.nilable(T.any(String, Pathname))], f: IO).void }
  def self.dump(env, f = $stdout)
    keys = self.keys(env)
    keys -= %w[CC CXX OBJC OBJCXX] if env["CC"] == env["HOMEBREW_CC"]

    keys.each do |key|
      value = env.fetch(key)

      s = +"#{key}: #{value}"
      case key
      when "CC", "CXX", "LD"
        s << " => #{Pathname.new(value).realpath}" if value.present? && File.symlink?(value)
      end
      s.freeze
      f.puts s
    end
  end
end
