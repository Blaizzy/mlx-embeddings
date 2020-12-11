# typed: true
# frozen_string_literal: true

module Homebrew
  # The {Service} class implements the DSL methods used in a formula's
  # `service` block and stores related instance variables. Most of these methods
  # also return the related instance variable when no argument is provided.
  class Service
    extend T::Sig

    RUN_TYPE_IMMEDIATE = "immediate"
    RUN_TYPE_INTERVAL = "interval"
    RUN_TYPE_CRON = "cron"

    # sig { params(formula: Formula).void }
    def initialize(formula)
      @formula = formula
      @run_type = RUN_TYPE_IMMEDIATE
      @environment_variables = {}
    end

    sig { params(command: T.nilable(T.any(T::Array[String], String, Pathname))).returns(T.nilable(Array)) }
    def run(command = nil)
      case T.unsafe(command)
      when nil
        @run
      when String, Pathname
        @run = [command]
      when Array
        @run = command
      else
        raise TypeError, "Service#run expects an Array"
      end
    end

    sig { params(path: T.nilable(T.any(String, Pathname))).returns(T.nilable(String)) }
    def working_dir(path = nil)
      case T.unsafe(path)
      when nil
        @working_dir
      when String, Pathname
        @working_dir = path.to_s
      else
        raise TypeError, "Service#working_dir expects a String"
      end
    end

    sig { params(path: T.nilable(T.any(String, Pathname))).returns(T.nilable(String)) }
    def log_path(path = nil)
      case T.unsafe(path)
      when nil
        @log_path
      when String, Pathname
        @log_path = path.to_s
      else
        raise TypeError, "Service#log_path expects a String"
      end
    end

    sig { params(path: T.nilable(T.any(String, Pathname))).returns(T.nilable(String)) }
    def error_log_path(path = nil)
      case T.unsafe(path)
      when nil
        @error_log_path
      when String, Pathname
        @error_log_path = path.to_s
      else
        raise TypeError, "Service#error_log_path expects a String"
      end
    end

    sig { params(value: T.nilable(T::Boolean)).returns(T.nilable(T::Boolean)) }
    def keep_alive(value = nil)
      case T.unsafe(value)
      when nil
        @keep_alive
      when true, false
        @keep_alive = value
      else
        raise TypeError, "Service#keep_alive expects a Boolean"
      end
    end

    sig { params(type: T.nilable(T.any(String, Symbol))).returns(T.nilable(String)) }
    def run_type(type = nil)
      case T.unsafe(type)
      when nil
        @run_type
      when "immediate", :immediate
        @run_type = type.to_s
      when RUN_TYPE_INTERVAL, RUN_TYPE_CRON
        raise TypeError, "Service#run_type does not support timers"
      when String
        raise TypeError, "Service#run_type allows: '#{RUN_TYPE_IMMEDIATE}'/'#{RUN_TYPE_INTERVAL}'/'#{RUN_TYPE_CRON}'"
      else
        raise TypeError, "Service#run_type expects a string"
      end
    end

    sig { params(variables: T.nilable(T::Hash[String, String])).returns(T.nilable(T::Hash[String, String])) }
    def environment_variables(variables = {})
      case T.unsafe(variables)
      when nil
        @environment_variables
      when Hash
        @environment_variables = variables
      else
        raise TypeError, "Service#environment_variables expects a hash"
      end
    end

    # The directory where the formula's variable files should be installed.
    # This directory is not inside the `HOMEBREW_CELLAR` so it persists
    # across upgrades.
    sig { returns(Pathname) }
    def var
      HOMEBREW_PREFIX/"var"
    end

    # The directory where the formula's configuration files should be installed.
    # Anything using `etc.install` will not overwrite other files on e.g. upgrades
    # but will write a new file named `*.default`.
    # This directory is not inside the `HOMEBREW_CELLAR` so it persists
    # across upgrades.
    sig { returns(Pathname) }
    def etc
      HOMEBREW_PREFIX/"etc"
    end

    # The directory where the formula's binaries should be installed.
    # This is symlinked into `HOMEBREW_PREFIX` after installation or with
    # `brew link` for formulae that are not keg-only.
    sig { returns(Pathname) }
    def opt_bin
      opt_prefix/"bin"
    end

    # The directory where the formula's binaries should be installed.
    # This is symlinked into `HOMEBREW_PREFIX` after installation or with
    # `brew link` for formulae that are not keg-only.
    sig { returns(Pathname) }
    def opt_sbin
      opt_prefix/"sbin"
    end

    # A stable path for this formula, when installed. Contains the formula name
    # but no version number. Only the active version will be linked here if
    # multiple versions are installed.
    sig { returns(Pathname) }
    def opt_prefix
      HOMEBREW_PREFIX/"opt/#{@formula.name}"
    end

    sig { returns(String) }
    def std_service_path_env
      "#{HOMEBREW_PREFIX}/bin:#{HOMEBREW_PREFIX}/sbin:/usr/bin:/bin:/usr/sbin:/sbin"
    end

    # Returns a `String` plist.
    # @return [String]
    sig { returns(String) }
    def to_plist
      clean_command = @run.select {|i| i.is_a?(Pathname)}
                          .map(&:to_s)

      base = {
        Label:            @formula.plist_name,
        RunAtLoad:        @run_type == RUN_TYPE_IMMEDIATE,
        ProgramArguments: clean_command,
      }

      base[:KeepAlive] = @keep_alive if @keep_alive == true
      base[:WorkingDirectory] = @working_dir if @working_dir.present?
      base[:StandardOutPath] = @log_path if @log_path.present?
      base[:StandardErrorPath] = @error_log_path if @error_log_path.present?
      base[:EnvironmentVariables] = @environment_variables unless @environment_variables.empty?

      base.to_plist
    end
  end
end
