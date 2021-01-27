# typed: false
# frozen_string_literal: true

require "cask/cmd"
require "completions"

# Helper functions for commands.
#
# @api private
module Commands
  module_function

  HOMEBREW_CMD_PATH = (HOMEBREW_LIBRARY_PATH/"cmd").freeze
  HOMEBREW_DEV_CMD_PATH = (HOMEBREW_LIBRARY_PATH/"dev-cmd").freeze
  HOMEBREW_INTERNAL_COMMAND_ALIASES = {
    "ls"          => "list",
    "homepage"    => "home",
    "-S"          => "search",
    "up"          => "update",
    "ln"          => "link",
    "instal"      => "install", # gem does the same
    "uninstal"    => "uninstall",
    "rm"          => "uninstall",
    "remove"      => "uninstall",
    "configure"   => "diy",
    "abv"         => "info",
    "dr"          => "doctor",
    "--repo"      => "--repository",
    "environment" => "--env",
    "--config"    => "config",
    "-v"          => "--version",
    "tc"          => "typecheck",
  }.freeze

  def valid_internal_cmd?(cmd)
    require?(HOMEBREW_CMD_PATH/cmd)
  end

  def valid_internal_dev_cmd?(cmd)
    require?(HOMEBREW_DEV_CMD_PATH/cmd)
  end

  def method_name(cmd)
    cmd.to_s
       .tr("-", "_")
       .downcase
       .to_sym
  end

  def args_method_name(cmd_path)
    cmd_path_basename = basename_without_extension(cmd_path)
    cmd_method_prefix = method_name(cmd_path_basename)
    "#{cmd_method_prefix}_args".to_sym
  end

  def internal_cmd_path(cmd)
    [
      HOMEBREW_CMD_PATH/"#{cmd}.rb",
      HOMEBREW_CMD_PATH/"#{cmd}.sh",
    ].find(&:exist?)
  end

  def internal_dev_cmd_path(cmd)
    [
      HOMEBREW_DEV_CMD_PATH/"#{cmd}.rb",
      HOMEBREW_DEV_CMD_PATH/"#{cmd}.sh",
    ].find(&:exist?)
  end

  # Ruby commands which can be `require`d without being run.
  def external_ruby_v2_cmd_path(cmd)
    path = which("#{cmd}.rb", Tap.cmd_directories)
    path if require?(path)
  end

  # Ruby commands which are run by being `require`d.
  def external_ruby_cmd_path(cmd)
    which("brew-#{cmd}.rb", PATH.new(ENV["PATH"]).append(Tap.cmd_directories))
  end

  def external_cmd_path(cmd)
    which("brew-#{cmd}", PATH.new(ENV["PATH"]).append(Tap.cmd_directories))
  end

  def path(cmd)
    internal_cmd = HOMEBREW_INTERNAL_COMMAND_ALIASES.fetch(cmd, cmd)
    path ||= internal_cmd_path(internal_cmd)
    path ||= internal_dev_cmd_path(internal_cmd)
    path ||= external_ruby_v2_cmd_path(cmd)
    path ||= external_ruby_cmd_path(cmd)
    path ||= external_cmd_path(cmd)
    path
  end

  def commands(external: true, aliases: false)
    cmds = internal_commands
    cmds += internal_developer_commands
    cmds += external_commands if external
    cmds += internal_commands_aliases if aliases
    cmds += cask_commands(aliases: aliases).map { |cmd| "cask #{cmd}" }
    cmds.sort
  end

  def internal_commands_paths(cask: true)
    cmds = find_commands HOMEBREW_CMD_PATH
    # can be removed when cask commands are removed and no longer odeprecated/odisabled
    cmds.delete(HOMEBREW_CMD_PATH/"cask.rb") unless cask
    cmds
  end

  def internal_developer_commands_paths
    find_commands HOMEBREW_DEV_CMD_PATH
  end

  def official_external_commands_paths(quiet:)
    %w[bundle services test-bot].map do |cmd|
      tap = Tap.fetch("Homebrew/#{cmd}")
      tap.install(quiet: quiet) unless tap.installed?
      external_ruby_v2_cmd_path(cmd)
    end
  end

  def internal_commands
    find_internal_commands(HOMEBREW_CMD_PATH).map(&:to_s)
  end

  def internal_developer_commands
    find_internal_commands(HOMEBREW_DEV_CMD_PATH).map(&:to_s)
  end

  def internal_commands_aliases
    HOMEBREW_INTERNAL_COMMAND_ALIASES.keys
  end

  def find_internal_commands(path)
    find_commands(path).map(&:basename)
                       .map(&method(:basename_without_extension))
  end

  def external_commands
    Tap.cmd_directories.flat_map do |path|
      find_commands(path).select(&:executable?)
                         .map(&method(:basename_without_extension))
                         .map { |p| p.to_s.delete_prefix("brew-").strip }
    end.map(&:to_s)
       .sort
  end

  def cask_commands(aliases: false)
    cmds = cask_internal_commands
    cmds += cask_internal_command_aliases if aliases
    cmds += cask_external_commands
    cmds
  end

  def cask_internal_commands
    Cask::Cmd.commands
  end

  def cask_internal_command_aliases
    Cask::Cmd.aliases.keys
  end

  def cask_external_commands
    PATH.new(Tap.cmd_directories, ENV["HOMEBREW_PATH"]).flat_map do |search_path|
      find_commands(search_path).map do |possible_command|
        path = possible_command.to_path
        command_name = path.match(/brewcask-(.*)\.rb/) { |data| data[1].delete_suffix(".rb") }
        if command_name.blank? && possible_command.executable?
          command_name = path.match(/brewcask-(.*)/) { |data| data[1] }
        end
        command_name
      end.compact
    end
  end

  def basename_without_extension(path)
    path.basename(path.extname)
  end

  def find_commands(path)
    Pathname.glob("#{path}/*")
            .select(&:file?)
            .sort
  end

  def rebuild_internal_commands_completion_list
    cmds = internal_commands + internal_developer_commands + internal_commands_aliases
    cmds.reject! { |cmd| Homebrew::Completions::COMPLETIONS_EXCLUSION_LIST.include? cmd }

    file = HOMEBREW_REPOSITORY/"completions/internal_commands_list.txt"
    file.atomic_write("#{cmds.sort.join("\n")}\n")
  end

  def rebuild_commands_completion_list
    # Ensure that the cache exists so we can build the commands list
    HOMEBREW_CACHE.mkpath

    cmds = commands(aliases: true).reject do |cmd|
      # TODO: (2.8) remove the cask check when `brew cask` is removed
      cmd.start_with?("cask ") || Homebrew::Completions::COMPLETIONS_EXCLUSION_LIST.include?(cmd)
    end

    all_commands_file = HOMEBREW_CACHE/"all_commands_list.txt"
    external_commands_file = HOMEBREW_CACHE/"external_commands_list.txt"
    all_commands_file.atomic_write("#{cmds.sort.join("\n")}\n")
    external_commands_file.atomic_write("#{external_commands.sort.join("\n")}\n")
  end

  def command_options(command)
    path = self.path(command)
    return if path.blank?

    if cmd_parser = Homebrew::CLI::Parser.from_cmd_path(path)
      cmd_parser.processed_options.map do |short, long, _, desc|
        [long || short, desc]
      end
    else
      options = []
      comment_lines = path.read.lines.grep(/^#:/)
      return options if comment_lines.empty?

      # skip the comment's initial usage summary lines
      comment_lines.slice(2..-1).each do |line|
        if / (?<option>-[-\w]+) +(?<desc>.*)$/ =~ line
          options << [option, desc]
        end
      end
      options
    end
  end

  def command_description(command, short: false)
    path = self.path(command)
    return if path.blank?

    if cmd_parser = Homebrew::CLI::Parser.from_cmd_path(path)
      if short
        cmd_parser.description.split(".").first
      else
        cmd_parser.description
      end
    else
      comment_lines = path.read.lines.grep(/^#:/)

      # skip the comment's initial usage summary lines
      comment_lines.slice(2..-1)&.each do |line|
        if /^#:  (?<desc>\w.*+)$/ =~ line
          return desc.split(".").first if short

          return desc
        end
      end
    end
  end

  def named_args_type(command)
    path = self.path(command)
    return if path.blank?

    cmd_parser = Homebrew::CLI::Parser.from_cmd_path(path)
    return if cmd_parser.blank?

    Array(cmd_parser.named_args_type)
  end

  # Returns the conflicts of a given `option` for `command`.
  def option_conflicts(command, option)
    path = self.path(command)
    return if path.blank?

    cmd_parser = Homebrew::CLI::Parser.from_cmd_path(path)
    return if cmd_parser.blank?

    cmd_parser.conflicts.map do |set|
      set.map! { |s| s.tr "_", "-" }
      set - [option] if set.include? option
    end.flatten.compact
  end
end
