## Define mini-templates for each portion of the doco.

<%!
  def indent(s, spaces=4):
      new = s.replace('\n', '\n' + ' ' * spaces)
      return ' ' * spaces + new.strip()

  def extract_param_descriptions(docstring):
      """Extract parameter descriptions from docstring."""
      param_desc = {}
      if docstring and 'Args:' in docstring:
          # Get the Args section
          args_section = docstring.split('Args:')[1]
          # Split at the next major section if it exists
          if '\n\n' in args_section:
              for section_start in ['Returns:', 'Raises:', 'Note:', 'Example:', 'Examples:']:
                  if section_start in args_section:
                      args_section = args_section.split(section_start)[0]

          lines = args_section.split('\n')
          current_param = None
          current_desc = []
          base_indent = None

          for line in lines:
              if not line.strip():
                  continue

              # Calculate the indentation level
              indent_level = len(line) - len(line.lstrip())

              # If this is a parameter definition line (contains "(type):" pattern)
              if '(' in line and '):' in line:
                  # If we were processing a previous parameter, save it
                  if current_param:
                      desc = ' '.join(current_desc)
                      # Add double line breaks after periods
                      desc = desc.replace('. ', '.<br/><br/>')
                      # Remove any trailing <br/><br/> if description ends with a period
                      if desc.endswith('.<br/><br/>'):
                          desc = desc[:-8] + '.'
                      param_desc[current_param] = desc

                  # Start new parameter
                  current_param = line.split('(')[0].strip()
                  desc_start = line.split('):')[1].strip()
                  current_desc = [desc_start] if desc_start else []
                  base_indent = indent_level
              else:
                  # If indentation is more than base, it's a continuation of the current description
                  if base_indent is not None and indent_level > base_indent and current_param:
                      current_desc.append(line.strip())

          # Save the last parameter
          if current_param:
              desc = ' '.join(current_desc)
              # Add double line breaks after periods
              desc = desc.replace('. ', '.<br/><br/>')
              # Remove any trailing <br/><br/> if description ends with a period
              if desc.endswith('.<br/><br/>'):
                  desc = desc[:-8] + '.'
              param_desc[current_param] = desc

      return param_desc

  def format_param_table(params, docstring):
      if not params:
          return ""
      param_descriptions = extract_param_descriptions(docstring)
      table = "| PARAMETER | DESCRIPTION |\n|--|--|\n"

      for param in params:
          # Split the parameter into name, type, and default
          parts = param.split(':')
          if len(parts) > 1:
              name = parts[0].strip()
              type_default = parts[1].strip()
              # Split type and default if exists
              if '=' in type_default:
                  type_val, default = type_default.split('=', 1)
                  type_val = type_val.strip().replace('|', '\\|')  # Escape pipe characters
                  default = default.strip().replace('|', '\\|')     # Escape pipe characters
              else:
                  type_val = type_default.replace('|', '\\|')      # Escape pipe characters
                  default = '-'
          else:
              name = param
              type_val = '-'
              default = '-'

          description = param_descriptions.get(name, "")
          formatted_desc = f"{description}<br/><br/>**TYPE:** `{type_val}`"
          if default != '-':
              formatted_desc += f"<br/><br/>**DEFAULT:** {default}"

          table += f"| `{name}` | {formatted_desc} |\n"

      return table
%>

<%def name="deflist(s)">
% if 'Args:' in s:
${indent(s.split('Args:')[0])}
% else:
${indent(s)}
% endif
</%def>

<%def name="h3(s)">### ${s}
</%def>

<%def name="h2(s)">## ${s}
</%def>

<%def name="function(func)" buffered="True">
<h3 id="${func.module.name}.${func.cls.name if func.cls else ''}.${func.name}" class="doc doc-heading">
<code class="doc-symbol doc-symbol-heading doc-symbol-method"></code>
<span class="doc doc-object-name doc-function-name">${func.name}</span>
<a href="#${func.module.name}.${func.cls.name if func.cls else ''}.${func.name}" class="headerlink" title="Permanent link"></a>
</h3>

<%
       returns = show_type_annotations and func.return_annotation() or ''
       if returns:
           returns = ' \N{non-breaking hyphen}> ' + returns
       params = func.params(annotate=show_type_annotations)
       if len(params) > 2:
           formatted_params = ',\n    '.join(params)
           signature = f"{func.name}(\n    {formatted_params}\n){returns}"
       else:
           signature = f"{func.name}({', '.join(params)}){returns}"
%>
```python
${signature}
```
${func.docstring | deflist}

% if len(params) > 0:
#### Parameters

${format_param_table(params, func.docstring)}
% endif
</%def>

<%def name="variable(var)" buffered="True">
<h3 id="${var.module.name}.${var.cls.name if var.cls else ''}.${var.name}" class="doc doc-heading">
<code class="doc-symbol doc-symbol-heading doc-symbol-attribute"></code>
<span class="doc doc-object-name doc-attribute-name">${var.name}</span>
<code class="doc-symbol doc-symbol-heading doc-symbol-${var.instance_var and 'instance' or 'class'}-attribute"></code>
<a href="#${var.module.name}.${var.cls.name if var.cls else ''}.${var.name}" class="headerlink" title="Permanent link"></a>
</h3>
<%
        annot = show_type_annotations and var.type_annotation() or ''
        if annot:
            annot = ': ' + annot
%>
${var.name}${annot}

${var.docstring | deflist}
</%def>

<%def name="class_(cls)" buffered="True">
    <h2 id="${cls.module.name}.${cls.name}" class="doc doc-heading">
        <code class="doc-symbol doc-symbol-heading doc-symbol-class"></code>
        <span class="doc doc-object-name doc-class-name">${cls.name}</span>
        <a href="#${cls.module.name}.${cls.name}" class="headerlink" title="Permanent link"></a>
    </h2>

<%
   params = cls.params(annotate=show_type_annotations)
   if len(params) > 2:
       formatted_params = ',\n    '.join(params)
       signature = f"{cls.name}(\n    {formatted_params}\n)"
   else:
       signature = f"{cls.name}({', '.join(params)})"
%>
```python
${signature}
```
${cls.docstring | deflist}

% if len(params) > 0:
### Parameters

${format_param_table(params, cls.docstring)}
% endif

<%
  class_vars = cls.class_variables(show_inherited_members, sort=sort_identifiers)
  static_methods = cls.functions(show_inherited_members, sort=sort_identifiers)
  inst_vars = cls.instance_variables(show_inherited_members, sort=sort_identifiers)
  methods = cls.methods(show_inherited_members, sort=sort_identifiers)
  mro = cls.mro()
  subclasses = cls.subclasses()
%>
% if class_vars:
    % for v in class_vars:
${variable(v)}

    % endfor
% endif
% if static_methods:
    % for f in static_methods:
${function(f)}

    % endfor
% endif
% if inst_vars:
    % for v in inst_vars:
${variable(v)}

    % endfor
% endif
% if methods:
    % for m in methods:
${function(m)}

    % endfor
% endif
</%def>

## Start the output logic for an entire module.

<%
  variables = module.variables(sort=sort_identifiers)
  classes = module.classes(sort=sort_identifiers)
  functions = module.functions(sort=sort_identifiers)
  submodules = module.submodules()
  heading = 'Namespace' if module.is_namespace else 'Module'
  symbol_name = module.name.split('.')[-1]
%>

---
sidebarTitle: ${symbol_name}
title: ${module.name}
---

% if submodules:
${h2('Sub-modules')}
    % for m in submodules:
* ${m.name}
    % endfor
% endif

% if variables:
${h2('Variables')}
    % for v in variables:
${variable(v)}

    % endfor
% endif

% if functions:
${h2('Functions')}
    % for f in functions:
${function(f)}

    % endfor
% endif

% if classes:
    % for c in classes:
${class_(c)}

    % endfor
% endif
