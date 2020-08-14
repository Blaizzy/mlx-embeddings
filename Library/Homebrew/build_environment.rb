# frozen_string_literal: true

# Settings for the build environment.
#
# @api private
class BuildEnvironment
  def initialize(*settings)
    @settings = Set.new(*settings)
  end

  def merge(*args)
    @settings.merge(*args)
    self
  end

  def <<(o)
    @settings << o
    self
  end

  def std?
    @settings.include? :std
  end

  def userpaths?
    @settings.include? :userpaths
  end

  # DSL for specifying build environment settings.
  module DSL
    def env(*settings)
      @env ||= BuildEnvironment.new
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

  def self.keys(env)
    KEYS & env.keys
  end

  def self.dump(env, f = $stdout)
    keys = self.keys(env)
    keys -= %w[CC CXX OBJC OBJCXX] if env["CC"] == env["HOMEBREW_CC"]

    keys.each do |key|
      value = env[key]
      s = +"#{key}: #{value}"
      case key
      when "CC", "CXX", "LD"
        s << " => #{Pathname.new(value).realpath}" if File.symlink?(value)
      end
      s.freeze
      f.puts s
    end
  end
end
