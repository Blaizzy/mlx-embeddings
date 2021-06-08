# typed: true
# frozen_string_literal: true

module Homebrew
  # The {Service} class implements the DSL methods used in a formula's
  # `service` block and stores related instance variables. Most of these methods
  # also return the related instance variable when no argument is provided.
  class Service
    extend T::Sig
    extend Forwardable

    RUN_TYPE_IMMEDIATE = "immediate"
    RUN_TYPE_INTERVAL = "interval"
    RUN_TYPE_CRON = "cron"

    # sig { params(formula: Formula).void }
    def initialize(formula, &block)
      @formula = formula
      @run_type = RUN_TYPE_IMMEDIATE
      @environment_variables = {}
      @service_block = block
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
    def root_dir(path = nil)
      case T.unsafe(path)
      when nil
        @root_dir
      when String, Pathname
        @root_dir = path.to_s
      else
        raise TypeError, "Service#root_dir expects a String or Pathname"
      end
    end

    sig { params(path: T.nilable(T.any(String, Pathname))).returns(T.nilable(String)) }
    def input_path(path = nil)
      case T.unsafe(path)
      when nil
        @input_path
      when String, Pathname
        @input_path = path.to_s
      else
        raise TypeError, "Service#input_path expects a String or Pathname"
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

    sig { params(value: T.nilable(Integer)).returns(T.nilable(Integer)) }
    def restart_delay(value = nil)
      case T.unsafe(value)
      when nil
        @restart_delay
      when Integer
        @restart_delay = value
      else
        raise TypeError, "Service#restart_delay expects an Integer"
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

    sig { params(variables: T::Hash[String, String]).returns(T.nilable(T::Hash[String, String])) }
    def environment_variables(variables = {})
      case T.unsafe(variables)
      when Hash
        @environment_variables = variables.transform_values(&:to_s)
      else
        raise TypeError, "Service#environment_variables expects a hash"
      end
    end

    sig { params(value: T.nilable(T::Boolean)).returns(T.nilable(T::Boolean)) }
    def macos_legacy_timers(value = nil)
      case T.unsafe(value)
      when nil
        @macos_legacy_timers
      when true, false
        @macos_legacy_timers = value
      else
        raise TypeError, "Service#macos_legacy_timers expects a Boolean"
      end
    end

    delegate [:bin, :etc, :libexec, :opt_bin, :opt_libexec, :opt_pkgshare, :opt_prefix, :opt_sbin, :var] => :@formula

    sig { returns(String) }
    def std_service_path_env
      "#{HOMEBREW_PREFIX}/bin:#{HOMEBREW_PREFIX}/sbin:/usr/bin:/bin:/usr/sbin:/sbin"
    end

    sig { returns(T::Array[String]) }
    def command
      instance_eval(&@service_block)
      @run.map(&:to_s)
    end

    sig { returns(String) }
    def manual_command
      instance_eval(&@service_block)
      vars = @environment_variables.except(:PATH)
                                   .map { |k, v| "#{k}=\"#{v}\"" }

      out = vars + command
      out.join(" ")
    end

    # Returns a `String` plist.
    # @return [String]
    sig { returns(String) }
    def to_plist
      base = {
        Label:            @formula.plist_name,
        RunAtLoad:        @run_type == RUN_TYPE_IMMEDIATE,
        ProgramArguments: command,
      }

      base[:KeepAlive] = @keep_alive if @keep_alive == true
      base[:LegacyTimers] = @macos_legacy_timers if @macos_legacy_timers == true
      base[:TimeOut] = @restart_delay if @restart_delay.present?
      base[:WorkingDirectory] = @working_dir if @working_dir.present?
      base[:RootDirectory] = @root_dir if @root_dir.present?
      base[:StandardInPath] = @input_path if @input_path.present?
      base[:StandardOutPath] = @log_path if @log_path.present?
      base[:StandardErrorPath] = @error_log_path if @error_log_path.present?
      base[:EnvironmentVariables] = @environment_variables unless @environment_variables.empty?

      base.to_plist
    end

    # Returns a `String` systemd unit.
    # @return [String]
    sig { returns(String) }
    def to_systemd_unit
      unit = <<~EOS
        [Unit]
        Description=Homebrew generated unit for #{@formula.name}

        [Install]
        WantedBy=multi-user.target

        [Service]
        Type=simple
        ExecStart=#{command.join(" ")}
      EOS

      options = []
      options << "Restart=always" if @keep_alive == true
      options << "RestartSec=#{restart_delay}" if @restart_delay.present?
      options << "WorkingDirectory=#{@working_dir}" if @working_dir.present?
      options << "RootDirectory=#{@root_dir}" if @root_dir.present?
      options << "StandardInput=file:#{@input_path}" if @input_path.present?
      options << "StandardOutput=append:#{@log_path}" if @log_path.present?
      options << "StandardError=append:#{@error_log_path}" if @error_log_path.present?
      options += @environment_variables.map { |k, v| "Environment=\"#{k}=#{v}\"" } if @environment_variables.present?

      unit + options.join("\n")
    end
  end
end
