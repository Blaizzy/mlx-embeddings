# typed: false
# frozen_string_literal: true

require "override"

module OnSystem
  extend T::Sig

  ARCH_OPTIONS = [:intel, :arm].freeze
  BASE_OS_OPTIONS = [:macos, :linux].freeze

  module_function

  sig { params(arch: Symbol).returns(T::Boolean) }
  def arch_condition_met?(arch)
    raise ArgumentError, "Invalid arch condition: #{arch.inspect}" unless ARCH_OPTIONS.include?(arch)

    current_arch = Homebrew::Override.arch || Hardware::CPU.type
    arch == current_arch
  end

  sig { params(os_name: Symbol, or_condition: T.nilable(Symbol)).returns(T::Boolean) }
  def os_condition_met?(os_name, or_condition = nil)
    if Homebrew::EnvConfig.simulate_macos_on_linux?
      return false if os_name == :linux
      return true if [:macos, *MacOS::Version::SYMBOLS.keys].include?(os_name)
    end

    if BASE_OS_OPTIONS.include?(os_name)
      if Homebrew::Override.none?
        return OS.linux? if os_name == :linux
        return OS.mac? if os_name == :macos
      end

      return Homebrew::Override.send("#{os_name}?")
    end

    raise ArgumentError, "Invalid OS condition: #{os_name.inspect}" unless MacOS::Version::SYMBOLS.key?(os_name)

    if or_condition.present? && [:or_newer, :or_older].exclude?(or_condition)
      raise ArgumentError, "Invalid OS `or_*` condition: #{or_condition.inspect}"
    end

    base_os = MacOS::Version.from_symbol(os_name)
    current_os = MacOS::Version.from_symbol(Homebrew::Override.os || MacOS.version.to_sym)

    return current_os >= base_os if or_condition == :or_newer
    return current_os <= base_os if or_condition == :or_older

    current_os == base_os
  end

  sig { params(method_name: Symbol).returns(Symbol) }
  def condition_from_method_name(method_name)
    method_name.to_s.sub(/^on_/, "").to_sym
  end

  sig { params(onto: Class, include_linux: T::Boolean).void }
  def setup_methods!(onto:, include_linux: true)
    ARCH_OPTIONS.each do |arch|
      onto.define_method("on_#{arch}") do |&block|
        @on_system_blocks_exist = true

        return unless OnSystem.arch_condition_met? OnSystem.condition_from_method_name(__method__)

        @called_in_on_system_block = true
        result = block.call
        @called_in_on_system_block = false

        result
      end
    end

    if include_linux
      BASE_OS_OPTIONS.each do |base_os|
        onto.define_method("on_#{base_os}") do |&block|
          @on_system_blocks_exist = true

          return unless OnSystem.os_condition_met? OnSystem.condition_from_method_name(__method__)

          @called_in_on_system_block = true
          result = block.call
          @called_in_on_system_block = false

          result
        end
      end
    end

    MacOS::Version::SYMBOLS.each_key do |os_name|
      onto.define_method("on_#{os_name}") do |or_condition = nil, &block|
        @on_system_blocks_exist = true

        os_condition = OnSystem.condition_from_method_name __method__
        return unless OnSystem.os_condition_met? os_condition, or_condition

        @called_in_on_system_block = true
        result = block.call
        @called_in_on_system_block = false

        result
      end
    end
  end
end
