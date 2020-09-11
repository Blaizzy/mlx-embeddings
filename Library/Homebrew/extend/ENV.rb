# frozen_string_literal: true

require "hardware"
require "diagnostic"
require "extend/ENV/shared"
require "extend/ENV/std"
require "extend/ENV/super"

def superenv?(env)
  env != "std" && Superenv.bin
end

module EnvActivation
  def activate_extensions!(env: nil)
    if superenv?(env)
      extend(Superenv)
    else
      extend(Stdenv)
    end
  end

  def with_build_environment(env: nil, cc: nil, build_bottle: false, bottle_arch: nil)
    old_env = to_hash.dup
    tmp_env = to_hash.dup.extend(EnvActivation)
    tmp_env.activate_extensions!(env: env)
    tmp_env.setup_build_environment(cc: cc, build_bottle: build_bottle, bottle_arch: bottle_arch)
    replace(tmp_env)
    yield
  ensure
    replace(old_env)
  end

  def sensitive?(key)
    /(cookie|key|token|password)/i =~ key
  end

  def sensitive_environment
    select { |key, _| sensitive?(key) }
  end

  def clear_sensitive_environment!
    each_key { |key| delete key if sensitive?(key) }
  end
end

ENV.extend(EnvActivation)
