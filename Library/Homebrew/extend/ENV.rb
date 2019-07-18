# frozen_string_literal: true

require "hardware"
require "extend/ENV/shared"
require "extend/ENV/std"
require "extend/ENV/super"

def superenv?
  ARGV.env != "std" && Superenv.bin
end

module EnvActivation
  def activate_extensions!
    if superenv?
      extend(Superenv)
    else
      extend(Stdenv)
    end
  end

  def with_build_environment
    old_env = to_hash.dup
    tmp_env = to_hash.dup.extend(EnvActivation)
    tmp_env.activate_extensions!
    tmp_env.setup_build_environment
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
