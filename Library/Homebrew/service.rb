# typed: true
# frozen_string_literal: true

module Homebrew
  # The {Service} class implements the DSL methods used in a formula's
  # `service` block and stores related instance variables. Most of these methods
  # also return the related instance variable when no argument is provided.
  class Service
    extend T::Sig
    extend Forwardable

    RUN_TYPE_IMMEDIATE = :immediate
    RUN_TYPE_INTERVAL = :interval
    RUN_TYPE_CRON = :cron

    PROCESS_TYPE_BACKGROUND = :background
    PROCESS_TYPE_STANDARD = :standard
    PROCESS_TYPE_INTERACTIVE = :interactive
    PROCESS_TYPE_ADAPTIVE = :adaptive

    KEEP_ALIVE_KEYS = [:always, :successful_exit, :crashed, :path].freeze

    # sig { params(formula: Formula).void }
    def initialize(formula, &block)
      @formula = formula
      @run_type = RUN_TYPE_IMMEDIATE
      @environment_variables = {}
      @service_block = block
    end

    sig { returns(Formula) }
    def f
      @formula
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

    sig {
      params(value: T.nilable(T.any(T::Boolean, T::Hash[Symbol, T.untyped])))
        .returns(T.nilable(T::Hash[Symbol, T.untyped]))
    }
    def keep_alive(value = nil)
      case T.unsafe(value)
      when nil
        @keep_alive
      when true, false
        @keep_alive = { always: value }
      when Hash
        hash = T.cast(value, Hash)
        unless (hash.keys - KEEP_ALIVE_KEYS).empty?
          raise TypeError, "Service#keep_alive allows only #{KEEP_ALIVE_KEYS}"
        end

        @keep_alive = value
      else
        raise TypeError, "Service#keep_alive expects a Boolean or Hash"
      end
    end

    sig { params(value: T.nilable(String)).returns(T.nilable(T::Hash[Symbol, String])) }
    def sockets(value = nil)
      case T.unsafe(value)
      when nil
        @sockets
      when String
        match = T.must(value).match(%r{([a-z]+)://([a-z0-9.]+):([0-9]+)}i)
        raise TypeError, "Service#sockets a formatted socket definition as <type>://<host>:<port>" if match.blank?

        type, host, port = match.captures
        @sockets = { host: host, port: port, type: type }
      else
        raise TypeError, "Service#sockets expects a String"
      end
    end

    # Returns a `Boolean` describing if a service is set to be kept alive.
    # @return [Boolean]
    sig { returns(T::Boolean) }
    def keep_alive?
      instance_eval(&@service_block)
      @keep_alive.present? && @keep_alive[:always] != false
    end

    sig { params(value: T.nilable(T::Boolean)).returns(T.nilable(T::Boolean)) }
    def launch_only_once(value = nil)
      case T.unsafe(value)
      when nil
        @launch_only_once
      when true, false
        @launch_only_once = value
      else
        raise TypeError, "Service#launch_only_once expects a Boolean"
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

    sig { params(value: T.nilable(Symbol)).returns(T.nilable(Symbol)) }
    def process_type(value = nil)
      case T.unsafe(value)
      when nil
        @process_type
      when :background, :standard, :interactive, :adaptive
        @process_type = value
      when Symbol
        raise TypeError, "Service#process_type allows: " \
                         "'#{PROCESS_TYPE_BACKGROUND}'/'#{PROCESS_TYPE_STANDARD}'/" \
                         "'#{PROCESS_TYPE_INTERACTIVE}'/'#{PROCESS_TYPE_ADAPTIVE}'"
      else
        raise TypeError, "Service#process_type expects a Symbol"
      end
    end

    sig { params(value: T.nilable(Symbol)).returns(T.nilable(Symbol)) }
    def run_type(value = nil)
      case T.unsafe(value)
      when nil
        @run_type
      when :immediate, :interval, :cron
        @run_type = value
      when Symbol
        raise TypeError, "Service#run_type allows: '#{RUN_TYPE_IMMEDIATE}'/'#{RUN_TYPE_INTERVAL}'/'#{RUN_TYPE_CRON}'"
      else
        raise TypeError, "Service#run_type expects a Symbol"
      end
    end

    sig { params(value: T.nilable(Integer)).returns(T.nilable(Integer)) }
    def interval(value = nil)
      case T.unsafe(value)
      when nil
        @interval
      when Integer
        @interval = value
      else
        raise TypeError, "Service#interval expects an Integer"
      end
    end

    sig { params(value: T.nilable(String)).returns(T.nilable(Hash)) }
    def cron(value = nil)
      case T.unsafe(value)
      when nil
        @cron
      when String
        @cron = parse_cron(T.must(value))
      else
        raise TypeError, "Service#cron expects a String"
      end
    end

    sig { returns(T::Hash[Symbol, T.any(Integer, String)]) }
    def default_cron_values
      {
        Month:   "*",
        Day:     "*",
        Weekday: "*",
        Hour:    "*",
        Minute:  "*",
      }
    end

    sig { params(cron_statement: String).returns(T::Hash[Symbol, T.any(Integer, String)]) }
    def parse_cron(cron_statement)
      parsed = default_cron_values

      case cron_statement
      when "@hourly"
        parsed[:Minute] = 0
      when "@daily"
        parsed[:Minute] = 0
        parsed[:Hour] = 0
      when "@weekly"
        parsed[:Minute] = 0
        parsed[:Hour] = 0
        parsed[:Weekday] = 0
      when "@monthly"
        parsed[:Minute] = 0
        parsed[:Hour] = 0
        parsed[:Day] = 1
      when "@yearly", "@annually"
        parsed[:Minute] = 0
        parsed[:Hour] = 0
        parsed[:Day] = 1
        parsed[:Month] = 1
      else
        cron_parts = cron_statement.split
        raise TypeError, "Service#parse_cron expects a valid cron syntax" if cron_parts.length != 5

        [:Minute, :Hour, :Day, :Month, :Weekday].each_with_index do |selector, index|
          parsed[selector] = Integer(cron_parts.fetch(index)) if cron_parts.fetch(index) != "*"
        end
      end

      parsed
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

    # Returns the `String` command to run manually instead of the service.
    # @return [String]
    sig { returns(String) }
    def manual_command
      instance_eval(&@service_block)
      vars = @environment_variables.except(:PATH)
                                   .map { |k, v| "#{k}=\"#{v}\"" }

      out = vars + command
      out.join(" ")
    end

    # Returns a `Boolean` describing if a service is timed.
    # @return [Boolean]
    sig { returns(T::Boolean) }
    def timed?
      instance_eval(&@service_block)
      @run_type == RUN_TYPE_CRON || @run_type == RUN_TYPE_INTERVAL
    end

    # Returns a `String` plist.
    # @return [String]
    sig { returns(String) }
    def to_plist
      # command needs to be first because it initializes all other values
      base = {
        Label:            @formula.plist_name,
        ProgramArguments: command,
        RunAtLoad:        @run_type == RUN_TYPE_IMMEDIATE,
      }

      base[:LaunchOnlyOnce] = @launch_only_once if @launch_only_once == true
      base[:LegacyTimers] = @macos_legacy_timers if @macos_legacy_timers == true
      base[:TimeOut] = @restart_delay if @restart_delay.present?
      base[:ProcessType] = @process_type.to_s.capitalize if @process_type.present?
      base[:StartInterval] = @interval if @interval.present? && @run_type == RUN_TYPE_INTERVAL
      base[:WorkingDirectory] = @working_dir if @working_dir.present?
      base[:RootDirectory] = @root_dir if @root_dir.present?
      base[:StandardInPath] = @input_path if @input_path.present?
      base[:StandardOutPath] = @log_path if @log_path.present?
      base[:StandardErrorPath] = @error_log_path if @error_log_path.present?
      base[:EnvironmentVariables] = @environment_variables unless @environment_variables.empty?

      if keep_alive?
        if (always = @keep_alive[:always].presence)
          base[:KeepAlive] = always
        elsif @keep_alive.key?(:successful_exit)
          base[:KeepAlive] = { SuccessfulExit: @keep_alive[:successful_exit] }
        elsif @keep_alive.key?(:crashed)
          base[:KeepAlive] = { Crashed: @keep_alive[:crashed] }
        elsif @keep_alive.key?(:path) && @keep_alive[:path].present?
          base[:KeepAlive] = { PathState: @keep_alive[:path].to_s }
        end
      end

      if @sockets.present?
        base[:Sockets] = {}
        base[:Sockets][:Listeners] = {
          SockNodeName:    @sockets[:host],
          SockServiceName: @sockets[:port],
          SockProtocol:    @sockets[:type].upcase,
          SockFamily:      "IPv4v6",
        }
      end

      if @cron.present? && @run_type == RUN_TYPE_CRON
        base[:StartCalendarInterval] = @cron.reject { |_, value| value == "*" }
      end

      # Adding all session types has as the primary effect that if you initialise it through e.g. a Background session
      # and you later "physically" sign in to the owning account (Aqua session), things shouldn't flip out.
      # Also, we're not checking @process_type here because that is used to indicate process priority and not
      # necessarily if it should run in a specific session type. Like database services could run with ProcessType
      # Interactive so they have no resource limitations enforced upon them, but they aren't really interactive in the
      # general sense.
      base[:LimitLoadToSessionType] = %w[Aqua Background LoginWindow StandardIO System]

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
      EOS

      # command needs to be first because it initializes all other values
      cmd = command.join(" ")

      options = []
      options << "Type=#{(@launch_only_once == true) ? "oneshot" : "simple"}"
      options << "ExecStart=#{cmd}"

      options << "Restart=always" if @keep_alive.present? && @keep_alive[:always].present?
      options << "RestartSec=#{restart_delay}" if @restart_delay.present?
      options << "WorkingDirectory=#{@working_dir}" if @working_dir.present?
      options << "RootDirectory=#{@root_dir}" if @root_dir.present?
      options << "StandardInput=file:#{@input_path}" if @input_path.present?
      options << "StandardOutput=append:#{@log_path}" if @log_path.present?
      options << "StandardError=append:#{@error_log_path}" if @error_log_path.present?
      options += @environment_variables.map { |k, v| "Environment=\"#{k}=#{v}\"" } if @environment_variables.present?

      unit + options.join("\n")
    end

    # Returns a `String` systemd unit timer.
    # @return [String]
    sig { returns(String) }
    def to_systemd_timer
      timer = <<~EOS
        [Unit]
        Description=Homebrew generated timer for #{@formula.name}

        [Install]
        WantedBy=timers.target

        [Timer]
        Unit=#{@formula.service_name}
      EOS

      instance_eval(&@service_block)
      options = []
      options << "Persistent=true" if @run_type == RUN_TYPE_CRON
      options << "OnUnitActiveSec=#{@interval}" if @run_type == RUN_TYPE_INTERVAL

      if @run_type == RUN_TYPE_CRON
        minutes = (@cron[:Minute] == "*") ? "*" : format("%02d", @cron[:Minute])
        hours   = (@cron[:Hour] == "*") ? "*" : format("%02d", @cron[:Hour])
        options << "OnCalendar=#{@cron[:Weekday]}-*-#{@cron[:Month]}-#{@cron[:Day]} #{hours}:#{minutes}:00"
      end

      timer + options.join("\n")
    end
  end
end
