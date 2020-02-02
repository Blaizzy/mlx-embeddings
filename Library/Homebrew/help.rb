# frozen_string_literal: true

HOMEBREW_HELP = <<~EOS
  Example usage:
    brew search [TEXT|/REGEX/]
    brew info [FORMULA...]
    brew install FORMULA...
    brew update
    brew upgrade [FORMULA...]
    brew uninstall FORMULA...
    brew list [FORMULA...]

  Troubleshooting:
    brew config
    brew doctor
    brew install --verbose --debug FORMULA

  Contributing:
    brew create [URL [--no-fetch]]
    brew edit [FORMULA...]

  Further help:
    brew commands
    brew help [COMMAND]
    man brew
    https://docs.brew.sh
EOS

# NOTE Keep the length of vanilla --help less than 25 lines!
# This is because the default Terminal height is 25 lines. Scrolling sucks
# and concision is important. If more help is needed we should start
# specialising help like the gem command does.
# NOTE Keep lines less than 80 characters! Wrapping is just not cricket.
# NOTE The reason the string is at the top is so 25 lines is easy to measure!

require "cli/parser"
require "commands"

module Homebrew
  module Help
    module_function

    def help(cmd = nil, flags = {})
      # Resolve command aliases and find file containing the implementation.
      if cmd
        cmd = HOMEBREW_INTERNAL_COMMAND_ALIASES.fetch(cmd, cmd)
        path = Commands.path(cmd)
        path ||= which("brew-#{cmd}")
        path ||= which("brew-#{cmd}.rb")
      end

      # Display command-specific (or generic) help in response to `UsageError`.
      if (error_message = flags[:usage_error])
        $stderr.puts path ? command_help(path) : HOMEBREW_HELP
        $stderr.puts
        onoe error_message
        exit 1
      end

      # Handle `brew` (no arguments).
      if flags[:empty_argv]
        $stderr.puts HOMEBREW_HELP
        exit 1
      end

      # Handle `brew (-h|--help|--usage|-?|help)` (no other arguments).
      if cmd.nil?
        puts HOMEBREW_HELP
        exit 0
      end

      # Resume execution in `brew.rb` for unknown commands.
      return if path.nil?

      # Display help for internal command (or generic help if undocumented).
      puts command_help(path)
      exit 0
    end

    def command_help(path)
      # Let OptionParser generate help text for commands which have a parser defined
      cmd = path.basename(path.extname)
      cmd_args_method_name = "#{cmd.to_s.tr("-", "_")}_args".to_sym
      begin
        return Homebrew.send(cmd_args_method_name)
                       .generate_help_text
      rescue NoMethodError => e
        raise if e.name != cmd_args_method_name

        nil
      end

      help_lines = command_help_lines(path)
      if help_lines.empty?
        opoo "No help text in: #{path}" if ARGV.homebrew_developer?
        HOMEBREW_HELP
      else
        Formatter.wrap(help_lines.join.gsub(/^  /, ""), COMMAND_DESC_WIDTH)
                 .sub("@hide_from_man_page ", "")
                 .sub(/^\* /, "#{Tty.bold}Usage: brew#{Tty.reset} ")
                 .gsub(/`(.*?)`/m, "#{Tty.bold}\\1#{Tty.reset}")
                 .gsub(%r{<([^\s]+?://[^\s]+?)>}) { |url| Formatter.url(url) }
                 .gsub(/<(.*?)>/m, "#{Tty.underline}\\1#{Tty.reset}")
                 .gsub(/\*(.*?)\*/m, "#{Tty.underline}\\1#{Tty.reset}")
      end
    end
  end
end
