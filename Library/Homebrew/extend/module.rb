# typed: false
# frozen_string_literal: true

class Module
  def attr_rw(*attrs)
    file, line, = caller.first.split(":")
    line = line.to_i

    attrs.each do |attr|
      module_eval <<-EOS, file, line
        def #{attr}(val=nil)           # def prefix(val=nil)
          @#{attr} ||= nil             #   @prefix ||= nil
          return @#{attr} if val.nil?  #   return @prefix if val.nil?
          @#{attr} = val               #   @prefix = val
        end                            # end
      EOS
    end
  end
end
